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


ALTERNATIVES_SYSTEM_PROMPT = """Jesteś ekspertem od suplementów diety i zakupów online.
Znasz polskie i międzynarodowe sklepy z suplementami oraz europejskie normy RDA.

Odpowiadasz WYŁĄCZNIE w formacie JSON.

Na podstawie analizy suplementu, zaproponuj:
1. Tańsze zamienniki całego produktu z oceną podobieństwa składu
2. Możliwość zakupu osobnych składników (DIY stack)
3. Sklepy, gdzie szukać
4. Pokrycie dziennego zapotrzebowania (RDA/NRV) dla każdego składnika

Format:
{
  "cheaper_alternatives": [
    {
      "name": "nazwa zamiennika",
      "reason": "dlaczego jest tańszy/lepszy stosunek jakości do ceny",
      "search_query": "fraza do wyszukania",
      "stores": ["lista sklepów gdzie szukać"],
      "similarity_score": 85,
      "similarity_details": "krótki opis co pokrywa a czego brak (np. 'zawiera 8/10 składników aktywnych, brak witaminy K2')",
      "matching_ingredients": ["składnik1", "składnik2"],
      "missing_ingredients": ["składnik którego brak"]
    }
  ],
  "diy_stack": [
    {
      "ingredient": "nazwa składnika",
      "amount_needed": "dawka na porcję",
      "search_query": "fraza do wyszukania",
      "estimated_price_per_serving": "szacowana cena za porcję w PLN lub null",
      "notes": "uwagi (np. dostępność, popularne marki)"
    }
  ],
  "recommended_stores": [
    {
      "name": "nazwa sklepu",
      "url": "adres strony",
      "notes": "uwagi o sklepie"
    }
  ],
  "daily_values": [
    {
      "ingredient": "nazwa składnika",
      "amount_per_serving": "dawka w produkcie",
      "nrv_percent": 75,
      "nrv_note": "opcjonalna uwaga (np. 'wyższe zapotrzebowanie u kobiet w ciąży')",
      "gender_note": "opcjonalnie: różnica K vs M jeśli dotyczy"
    }
  ],
  "savings_potential": "opis potencjalnych oszczędności",
  "advice": "praktyczna rada dla użytkownika"
}

Zasady similarity_score:
- 100 = identyczny skład i dawki
- 80-99 = brakuje 1-2 mniej istotnych składników
- 60-79 = pokrywa główne składniki aktywne, różni się w pomocniczych
- 40-59 = podobna kategoria, ale znaczące różnice w składzie
- poniżej 40 = tylko częściowe podobieństwo

Dla daily_values używaj europejskich wartości NRV (Nutrient Reference Values) z rozporządzenia UE 1169/2011."""


def generate_alternatives(analysis: AnalysisResult, original_price: Optional[str] = None, user_profile: Optional[str] = None) -> dict:
    """Generate cheaper alternatives and DIY stack suggestions."""
    client = anthropic.Anthropic()

    profile_info = f"\nProfil użytkownika: {user_profile}" if user_profile else "\nProfil użytkownika: nie podano (użyj uniwersalnych wartości NRV)"

    content = f"""Analizowany suplement:
Nazwa: {analysis.product_name}
Marka: {analysis.brand or 'nieznana'}
Kategoria: {analysis.category}
Cena oryginalna: {original_price or 'nieznana'}{profile_info}

Składniki (kluczowe aktywne):
{', '.join(analysis.key_active_ingredients)}

Wszystkie składniki:
{json.dumps([i.model_dump() for i in analysis.ingredients], ensure_ascii=False, indent=2)}

Podsumowanie: {analysis.summary}

Zaproponuj tańsze zamienniki z similarity_score, DIY stack i daily_values dla polskiego użytkownika."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=ALTERNATIVES_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    response_text = message.content[0].text.strip()
    response_text = _clean_json(response_text)
    return json.loads(response_text)
