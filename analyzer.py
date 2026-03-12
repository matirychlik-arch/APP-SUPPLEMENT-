import anthropic
import json
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

    # Strip markdown code blocks if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    data = json.loads(response_text)
    return AnalysisResult(**data)


ALTERNATIVES_SYSTEM_PROMPT = """Jesteś ekspertem od suplementów diety i zakupów online.
Znasz polskie i międzynarodowe sklepy z suplementami.

Odpowiadasz WYŁĄCZNIE w formacie JSON.

Na podstawie analizy suplementu, zaproponuj:
1. Tańsze zamienniki całego produktu (inne marki z tym samym składem)
2. Możliwość zakupu osobnych składników (tzw. "stack" DIY)
3. Sklepy, gdzie szukać

Format:
{
  "cheaper_alternatives": [
    {
      "name": "nazwa zamiennika",
      "reason": "dlaczego jest tańszy/lepszy stosunek jakości do ceny",
      "search_query": "fraza do wyszukania",
      "stores": ["lista sklepów gdzie szukać"]
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
  "savings_potential": "opis potencjalnych oszczędności",
  "advice": "praktyczna rada dla użytkownika"
}"""


def generate_alternatives(analysis: AnalysisResult, original_price: Optional[str] = None) -> dict:
    """Generate cheaper alternatives and DIY stack suggestions."""
    client = anthropic.Anthropic()

    content = f"""Analizowany suplement:
Nazwa: {analysis.product_name}
Marka: {analysis.brand or 'nieznana'}
Kategoria: {analysis.category}
Cena oryginalna: {original_price or 'nieznana'}

Składniki (kluczowe aktywne):
{', '.join(analysis.key_active_ingredients)}

Wszystkie składniki:
{json.dumps([i.model_dump() for i in analysis.ingredients], ensure_ascii=False, indent=2)}

Podsumowanie: {analysis.summary}

Zaproponuj tańsze zamienniki i opcję DIY stack dla polskiego użytkownika."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2500,
        system=ALTERNATIVES_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    response_text = message.content[0].text.strip()

    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    return json.loads(response_text)
