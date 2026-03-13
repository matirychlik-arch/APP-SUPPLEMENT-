import anthropic
import json
import re
from typing import Optional
from scraper import ProductInfo
from pydantic import BaseModel


class Ingredient(BaseModel):
    name: str
    amount: Optional[str] = None
    unit: Optional[str] = None
    daily_value: Optional[str] = None
    notes: Optional[str] = None


class AnalysisResult(BaseModel):
    product_name: str
    brand: Optional[str] = None
    category: str
    ingredients: list[Ingredient]
    total_servings: Optional[str] = None
    serving_size: Optional[str] = None
    key_active_ingredients: list[str]
    summary: str
    search_queries: list[str]  # Suggested search queries for finding alternatives


SYSTEM_PROMPT = """Jesteś ekspertem od suplementów diety. Analizujesz skład suplementów i pomagasz użytkownikom znaleźć tańsze alternatywy.

Odpowiadasz WYŁĄCZNIE w formacie JSON. Nie dodawaj żadnego tekstu poza JSONem.

Twoje zadanie:
1. Przeanalizuj podane informacje o suplemencie
2. Wyodrębnij wszystkie składniki z dawkami
3. Zidentyfikuj kluczowe aktywne składniki (te, które nadają produktowi wartość)
4. Zaproponuj zapytania wyszukiwania do znalezienia tańszych alternatyw lub osobnych składników

Format odpowiedzi:
{
  "product_name": "nazwa produktu",
  "brand": "marka lub null",
  "category": "kategoria (np. pre-workout, białko, witaminy, kompleks witaminowy, omega-3, itp.)",
  "ingredients": [
    {
      "name": "nazwa składnika",
      "amount": "ilość lub null",
      "unit": "jednostka (mg, g, mcg, IU, itp.) lub null",
      "daily_value": "% dziennego zapotrzebowania lub null",
      "notes": "dodatkowe uwagi lub null"
    }
  ],
  "total_servings": "liczba porcji w opakowaniu lub null",
  "serving_size": "wielkość porcji lub null",
  "key_active_ingredients": ["lista kluczowych składników aktywnych"],
  "summary": "krótkie podsumowanie produktu po polsku (2-3 zdania)",
  "search_queries": [
    "zapytanie do wyszukiwania tańszego zamiennika 1",
    "zapytanie do wyszukiwania składnika 1 osobno",
    "zapytanie do wyszukiwania składnika 2 osobno"
  ]
}"""


def analyze_supplement(product: ProductInfo) -> AnalysisResult:
    """Use Claude to analyze supplement ingredients and generate search queries."""
    client = anthropic.Anthropic()

    content_parts = [f"Produkt: {product.name}"]
    if product.brand:
        content_parts.append(f"Marka: {product.brand}")
    if product.price:
        content_parts.append(f"Cena: {product.price} {product.currency}")
    if product.ingredients_raw:
        content_parts.append(f"\nSkład/Supplement Facts:\n{product.ingredients_raw}")
    elif product.description:
        content_parts.append(f"\nOpis produktu:\n{product.description}")

    user_message = "\n".join(content_parts)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = message.content[0].text.strip()
    response_text = _clean_json(response_text)
    data = json.loads(response_text)
    return AnalysisResult(**data)


def _clean_json(text: str) -> str:
    """Strip markdown fences, comments, and trailing commas from JSON string."""
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        text = match.group(1)

    # Remove JS-style single-line comments
    text = re.sub(r"//[^\n\"]*\n", "\n", text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


ALTERNATIVES_SYSTEM_PROMPT = """Jesteś ekspertem od suplementów diety i zakupów online w Polsce.
Znasz ceny suplementów w polskich i międzynarodowych sklepach (iHerb, Myprotein, Allegro, Amazon, sklepy PL).

Jeśli podano profil użytkownika (płeć, wiek, aktywność), PERSONALIZUJ rekomendacje:
- Sportowiec: preferuj wyższe dawki elektrolitów/kreatyny/BCAA, zaznacz w advice
- Kobieta: zwróć uwagę na żelazo, kwas foliowy, magnez
- 51+: wit. D3+K2 ważne, collagen, wit. B12
- Uwzględnij profil w polu "advice" – napisz co jest szczególnie ważne dla tej osoby

Odpowiadasz WYŁĄCZNIE w formacie JSON. Bądź zwięzły – max 3 alternatywy, max 6 składników DIY.

WAŻNE: Dla każdej alternatywy podaj szacunkową cenę w PLN. Dla każdego składnika DIY podaj:
- estimated_price_package: szacunkowa cena opakowania w PLN (np. "45 PLN")
- estimated_servings_in_package: ile porcji w opakowaniu (np. 60)
- estimated_price_per_serving: cena za 1 porcję = package/servings (np. "0.75 PLN")

Format:
{
  "cheaper_alternatives": [
    {
      "name": "nazwa zamiennika",
      "reason": "krótki powód (1 zdanie)",
      "search_query": "fraza do wyszukania",
      "stores": ["sklep1", "sklep2"],
      "estimated_price": "89 PLN",
      "similarity_score": 85,
      "similarity_details": "krótki opis (1 zdanie)",
      "matching_ingredients": ["składnik1"],
      "missing_ingredients": ["brakujący"]
    }
  ],
  "diy_stack": [
    {
      "ingredient": "nazwa",
      "amount_needed": "dawka na porcję",
      "search_query": "fraza",
      "estimated_price_package": "35 PLN",
      "estimated_servings_in_package": 60,
      "estimated_price_per_serving": "0.58 PLN",
      "notes": "krótka uwaga"
    }
  ],
  "recommended_stores": [
    {"name": "nazwa", "url": "https://...", "notes": "uwaga"}
  ],
  "savings_potential": "1-2 zdania o oszczędnościach",
  "advice": "1-2 zdania porady"
}

similarity_score: 100=identyczny, 80-99=brak 1-2 składników, 60-79=główne OK różni się w dodatkach, <60=częściowe podobieństwo.
Ceny są szacunkowe – zaznacz to w advice."""

DAILY_VALUES_SYSTEM_PROMPT = """Jesteś dietetykiem. Znasz europejskie normy NRV (rozporządzenie UE 1169/2011) oraz zalecenia dla różnych grup.

Odpowiadasz WYŁĄCZNIE w formacie JSON. Podaj NRV tylko dla składników które mają ustalone normy UE.

WAŻNE: Jeśli podano profil użytkownika (płeć, wiek, aktywność), DOSTOSUJ nrv_percent do jego indywidualnych potrzeb:
- Sportowcy/wysoka aktywność: magnez +20-30%, wit. B1/B2/B3/B6 +20%, żelazo bez zmian (M) lub +50% (K sportowiec)
- Kobiety: żelazo 150% wyższa norma niż mężczyźni (18mg vs 10mg), kwas foliowy ważniejszy
- Wiek 51+: wit. D +50%, wit. B12 +20%, wapń +20%
- Wiek 18-30: normy bazowe
Jeśli profil nie podany, używaj standardowych norm UE.

Format:
{
  "daily_values": [
    {
      "ingredient": "nazwa składnika",
      "amount_per_serving": "dawka z produktu",
      "nrv_percent": 75,
      "nrv_note": "opcjonalna uwaga max 1 zdanie",
      "gender_note": "różnica K vs M jeśli istotna, max 1 zdanie lub null"
    }
  ]
}

Jeśli składnik nie ma normy NRV (np. ekstrakty ziołowe, aminokwasy bez normy), pomiń go."""


def _call_claude(system: str, content: str, max_tokens: int = 3000) -> dict:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    response_text = message.content[0].text.strip()
    try:
        return json.loads(_clean_json(response_text))
    except json.JSONDecodeError as e:
        import logging
        logging.getLogger(__name__).error(f"JSON parse error: {e}\nRaw response (first 500 chars): {response_text[:500]}")
        raise


def generate_alternatives(analysis: AnalysisResult, original_price: Optional[str] = None, user_profile: Optional[str] = None) -> dict:
    """Generate cheaper alternatives, DIY stack, and daily values via two separate calls."""
    profile_info = f"\nProfil użytkownika: {user_profile}" if user_profile else ""
    ingredients_summary = ", ".join(
        f"{i.name} {i.amount or ''}{i.unit or ''}".strip()
        for i in analysis.ingredients
    )

    base_content = f"""Suplement: {analysis.product_name} ({analysis.brand or 'nieznana marka'})
Kategoria: {analysis.category}
Cena: {original_price or 'nieznana'}{profile_info}
Kluczowe składniki aktywne: {', '.join(analysis.key_active_ingredients)}
Wszystkie składniki: {ingredients_summary}"""

    log = __import__('logging').getLogger(__name__)
    log.info(f"[generate_alternatives] profile={user_profile!r}")

    # Call 1: alternatives + DIY + stores
    result = _call_claude(ALTERNATIVES_SYSTEM_PROMPT, base_content + "\n\nZaproponuj tańsze zamienniki i DIY stack dla polskiego użytkownika.", max_tokens=4000)
    log.info(f"[generate_alternatives] call1 OK, keys={list(result.keys())}")

    # Call 2: daily values
    try:
        dv_result = _call_claude(
            DAILY_VALUES_SYSTEM_PROMPT,
            base_content + "\n\nPodaj pokrycie NRV dla składników z ustalonymi normami UE.",
            max_tokens=2500,
        )
        result["daily_values"] = dv_result.get("daily_values", [])
        log.info(f"[generate_alternatives] call2 OK, {len(result['daily_values'])} NRV entries")
    except Exception as e:
        log.warning(f"[generate_alternatives] call2 FAILED: {e}")
        result["daily_values"] = []

    return result
