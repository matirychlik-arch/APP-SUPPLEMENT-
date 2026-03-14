import anthropic
import json
import re
from typing import Optional
from scraper import ProductInfo
from pydantic import BaseModel, field_validator


class Ingredient(BaseModel):
    name: str
    amount: Optional[str] = None
    unit: Optional[str] = None
    daily_value: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("amount", "unit", "daily_value", "notes", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        return str(v) if v is not None else None


class AnalysisResult(BaseModel):
    product_name: str
    brand: Optional[str] = None
    category: str
    ingredients: list[Ingredient]
    total_servings: Optional[str] = None
    serving_size: Optional[str] = None
    key_active_ingredients: list[str]
    summary: str
    search_queries: list[str]

    @field_validator("total_servings", "serving_size", "brand", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        return str(v) if v is not None else None


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
  "total_servings": "liczba porcji w opakowaniu — WYMAGANE gdy możliwe do obliczenia. Przykłady: '60 kapsułek, 2/porcję' → '30'; 'na 30 dni' → '30'; 'Supplement Facts: 30 servings' → '30'. Podaj tylko liczbę jako string (np. '30'), lub null gdy brak danych.",
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
        temperature=0,
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
- Wyczynowy/5-7x tyg.: preferuj wyższe dawki elektrolitów/kreatyny/BCAA, zaznacz w advice
- Kobieta: zwróć uwagę na żelazo, kwas foliowy, magnez
- 45-54 lat: wit. D3+K2, magnez, profilaktyka stawów
- 55+ lat: wit. D3+K2, collagen, wit. B12, wapń
- Uwzględnij profil w polu "advice" – napisz co jest szczególnie ważne dla tej osoby

Odpowiadasz WYŁĄCZNIE w formacie JSON. Bądź zwięzły – max 3 alternatywy, max 6 składników DIY.

KLUCZOWE — DOBÓR ZAMIENNIKA:
Szukaj produktów które zawierają jak NAJWIĘKSZĄ LICZBĘ tych samych aktywnych składników co oryginał.
Kolejność priorytetów: (1) maksymalne pokrycie składników aktywnych → (2) niższa cena → (3) jakość form.
NIE proponuj produktu tylko dlatego że jest tani — musi pokrywać co najmniej 50% aktywnych składników oryginału.

Jak obliczyć similarity_score:
similarity_score = round(liczba_wspólnych_aktywnych_składników / łączna_liczba_aktywnych_składników_oryginału × 100)
Licz TYLKO składniki aktywne (nie substancje pomocnicze, nie formy chemiczne).
Przykład: oryginał ma 10 aktywnych składników, zamiennik pokrywa 8 → similarity_score = 80.

Sortuj cheaper_alternatives malejąco wg similarity_score — najbardziej podobny składem JAKO PIERWSZY.

WAŻNE: Dla każdej alternatywy ORAZ każdego składnika DIY podaj:
- estimated_price_package: szacunkowa cena opakowania w PLN (np. "45 PLN")
- estimated_servings_in_package: ile porcji w opakowaniu (np. 60)
- estimated_price_per_serving: cena za 1 porcję = package/servings (np. "0.75 PLN")
To kluczowe – porównujemy PORCJĘ DO PORCJI z oryginalnym produktem.

Format:
{
  "cheaper_alternatives": [
    {
      "name": "nazwa zamiennika",
      "reason": "krótki powód (1 zdanie)",
      "search_query": "fraza do wyszukania",
      "stores": ["sklep1", "sklep2"],
      "estimated_price": "89 PLN",
      "estimated_servings_in_package": 30,
      "estimated_price_per_serving": "2.97 PLN",
      "similarity_score": 85,
      "similarity_details": "krótki opis (1 zdanie)",
      "matching_ingredients": ["składnik1"],
      "missing_ingredients": ["brakujący"],
      "quality_score": 72,
      "quality_verdict": "Dobry",
      "vegan_note": null
    }
  ],
  "diy_stack": [
    {
      "ingredient": "nazwa składnika",
      "amount_needed": "dawka na porcję",
      "search_query": "fraza (wersja domyślna)",
      "estimated_price_package": "89 PLN",
      "estimated_servings_in_package": 30,
      "estimated_price_per_serving": "2.97 PLN",
      "notes": "krótka uwaga lub 'wersja wegańska'",
      "alternative_option": null
    }
  ],
  "diy_coverage_score": 85,
  "recommended_stores": [
    {"name": "nazwa", "url": "https://...", "notes": "uwaga"}
  ],
  "savings_potential": "1-2 zdania o oszczędnościach",
  "advice": "1-2 zdania porady"
}

similarity_score: procent aktywnych składników oryginału obecnych w zamienniku (0–100). 90–100=prawie identyczny, 70–89=brak 1-3 składników, 50–69=główne składniki pokryte, <50=tylko częściowe podobieństwo (unikaj jeśli możliwe).
Zawsze wypełnij matching_ingredients (lista wspólnych) i missing_ingredients (czego brakuje).
quality_score: szacunkowa jakość samego zamiennika (0-100). 80-100=Doskonały, 60-79=Dobry, 40-59=Przeciętny, <40=Słaby. Oceń formy składników, markę, wartość za cenę.
diy_coverage_score: ile % aktywnych składników oryginału pokrywa DIY stack (0-100). 100=wszystkie pokryte, 0=brak pokrycia.
vegan_note: TYLKO gdy alternatywa jest nieweganską wersją (np. tran zamiast alg, D3 z lanoliny zamiast z porostów). Ustaw na krótki string np. "Wersja nieweganiska — zwykle tańsza" lub null.
alternative_option: OPCJONALNE — używaj TYLKO gdy składnik DIY ma wyraźnie tańszą nieweganską wersję (np. omega-3 z alg vs tran rybny, D3 z porostów vs D3 z lanoliny). Struktura:
  { "name": "Tran rybny omega-3", "search_query": "...", "estimated_price_package": "35 PLN", "estimated_servings_in_package": 60, "estimated_price_per_serving": "0.58 PLN", "notes": "Wersja nieweganiska — 3-4× tańsza, identyczny efekt" }
  Dla większości składników ustaw alternative_option = null.
  Przy obliczaniu kosztów w diy_coverage_score uwzględnij najtańszą dostępną opcję.
Ceny są szacunkowe – zaznacz to w advice.

OPCJA NIEWEGANISKA w cheaper_alternatives: Jeśli produkt zawiera składniki w wersji wegańskiej (np. DHA/EPA z alg morskich, witamina D3 z porostów, wegańska K2), zaproponuj w cheaper_alternatives RÓWNIEŻ najtańszą nieweganską alternatywę (np. tran rybny/fish oil dla omega-3 algowego, D3 z lanoliny owczej). Ustaw "vegan_note" na krótki opis różnicy. Wyjaśnij różnicę w polu "reason". Umieść ją na liście alternatyw jeśli jest wyraźnie tańsza."""

DAILY_VALUES_SYSTEM_PROMPT = """Jesteś dietetykiem. Znasz europejskie normy NRV (rozporządzenie UE 1169/2011) oraz zalecenia dla różnych grup.

Odpowiadasz WYŁĄCZNIE w formacie JSON. Podaj NRV tylko dla składników które mają ustalone normy UE.

WAŻNE: Jeśli podano profil użytkownika (płeć, wiek, aktywność), DOSTOSUJ nrv_percent do jego indywidualnych potrzeb:
- Aktywny/wyczynowy (5-7x tyg.): magnez +20-30%, wit. B1/B2/B3/B6 +20%, żelazo bez zmian (M) lub +50% (K aktywna)
- Kobiety: żelazo 150% wyższa norma niż mężczyźni (18mg vs 10mg), kwas foliowy ważniejszy
- Wiek 45-54: wit. D +30%, wit. B12 +10%, wapń +10%
- Wiek 55+: wit. D +50%, wit. B12 +20%, wapń +20%, magnez +10%
- Wiek 18-24, 25-34: normy bazowe
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


QUALITY_SYSTEM_PROMPT = """Jesteś ekspertem farmaceutycznym i technologiem żywności. Specjalizujesz się w ocenie jakości form składników suplementów diety.

Odpowiadasz WYŁĄCZNIE w formacie JSON. Bądź zwięzły i konkretny.

Twoje zadanie:
1. Dla każdego składnika oceń jego formę chemiczną i jakość:
   - Rozpoznaj czy to forma premium (np. chelaty, methylowane witaminy) czy tania/gorsza (np. tlenki, cyanocobalamina)
   - Zidentyfikuj patentowane formy: Creapure®, Albion®/TRAACS®, Ferrochel®, Quatrefolic®, MenaQ7®, PureWay-C®, Carnipure®, HMB®, Bioperine®, Ashwagandha KSM-66®/Sensoril®, itp.
   - Oceń bioprzyswajalnością

2. Oblicz SupScore (0-100) jako ważoną ocenę:
   - ingredient_score (50%): jakość form składników, obecność patentowanych form
   - value_score (30%): wartość za cenę vs rynek (0=drogi za to co daje, 100=świetna wartość)
   - brand_score (20%): reputacja marki (znana marka premium=80-100, no-name=30-50, mała marka=50-70)

3. Jeśli dostępny rating ze strony, uwzględnij go jako dodatkowy sygnał dla value_score.

Progi SupScore:
- 80-100: "Doskonały" (premium formy, dobra wartość)
- 60-79: "Dobry" (mix form, przyzwoita wartość)
- 40-59: "Przeciętny" (głównie tanie formy lub wysoka cena za jakość)
- <40: "Słaby" (niskiej jakości formy, słaba wartość)

Format odpowiedzi:
{
  "ingredient_qualities": [
    {
      "name": "nazwa składnika",
      "form": "forma chemiczna (np. 'glicynian magnezu', 'methylcobalamina', 'tlenek magnezu')",
      "quality_tier": "premium|standard|niska",
      "bioavailability": "wysoka|średnia|niska",
      "branded_form": "Creapure® lub null",
      "quality_reason": "1 zdanie uzasadnienia"
    }
  ],
  "sup_score": 78,
  "ingredient_score": 82,
  "value_score": 65,
  "brand_score": 75,
  "verdict": "Dobry",
  "pros": ["zaleta 1", "zaleta 2"],
  "cons": ["wada 1", "wada 2"],
  "quality_summary": "2 zdania podsumowania jakości produktu po polsku"
}

Oceniaj tylko składniki aktywne (nie pomocnicze jak stearyn magnezu, celuloza, itp.). Maksymalnie 3 pros i 3 cons."""


def _call_claude(system: str, content: str, max_tokens: int = 3000) -> dict:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        temperature=0,
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


def analyze_quality(
    analysis: AnalysisResult,
    original_price: Optional[str] = None,
    rating: Optional[str] = None,
    review_count: Optional[str] = None,
    user_profile: Optional[str] = None,
) -> dict:
    """Analyze ingredient quality and calculate SupScore via one Claude call."""
    log = __import__('logging').getLogger(__name__)

    ingredients_summary = ", ".join(
        f"{i.name} {i.amount or ''}{i.unit or ''}".strip()
        for i in analysis.ingredients
    )
    content_parts = [
        f"Suplement: {analysis.product_name} ({analysis.brand or 'nieznana marka'})",
        f"Kategoria: {analysis.category}",
        f"Cena: {original_price or 'nieznana'}",
        f"Wszystkie składniki: {ingredients_summary}",
        f"Kluczowe składniki aktywne: {', '.join(analysis.key_active_ingredients)}",
    ]
    if rating:
        review_info = f"{rating}/5" + (f" ({review_count} opinii)" if review_count else "")
        content_parts.append(f"Ocena klientów: {review_info}")
    if user_profile:
        content_parts.append(f"Profil użytkownika: {user_profile}")

    content = "\n".join(content_parts)
    try:
        result = _call_claude(QUALITY_SYSTEM_PROMPT, content, max_tokens=3000)
        log.info(f"[analyze_quality] OK, sup_score={result.get('sup_score')}")
        return result
    except Exception as e:
        log.error(f"[analyze_quality] FAILED: {e}")
        return {
            "ingredient_qualities": [],
            "sup_score": None,
            "ingredient_score": None,
            "value_score": None,
            "brand_score": None,
            "verdict": None,
            "pros": [],
            "cons": [],
            "quality_summary": None,
        }


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
