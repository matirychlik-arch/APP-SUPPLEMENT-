import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import Optional
import re


class ProductInfo(BaseModel):
    url: str
    name: str
    brand: Optional[str] = None
    price: Optional[str] = None
    currency: Optional[str] = "PLN"
    ingredients_raw: Optional[str] = None
    description: Optional[str] = None
    serving_size: Optional[str] = None
    servings_per_container: Optional[str] = None


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _extract_price(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """Try common price selectors across multiple supplement stores."""
    price_selectors = [
        # Generic structured data
        'span[itemprop="price"]',
        'meta[itemprop="price"]',
        # Common class patterns
        ".price",
        ".product-price",
        ".current-price",
        ".sale-price",
        '[class*="price"]',
        # Polish stores
        ".cena",
        ".price-final",
    ]
    for sel in price_selectors:
        el = soup.select_one(sel)
        if el:
            content = el.get("content") or el.get_text(strip=True)
            # Extract numeric price
            match = re.search(r"[\d\s]+[,.]?\d*", content)
            if match:
                raw = match.group().replace(" ", "").replace(",", ".")
                # Detect currency
                currency = "PLN"
                text = el.get_text(" ", strip=True)
                if "$" in text or "USD" in text:
                    currency = "USD"
                elif "€" in text or "EUR" in text:
                    currency = "EUR"
                elif "£" in text or "GBP" in text:
                    currency = "GBP"
                return raw, currency
    return None, None


def _extract_ingredients(soup: BeautifulSoup) -> Optional[str]:
    """Extract supplement facts / ingredients section."""
    # Try to find tables with supplement facts
    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        if any(kw in text.lower() for kw in ["supplement facts", "składniki", "ingredients", "skład"]):
            return text[:3000]

    # Look for divs/sections with ingredient keywords
    keywords = [
        "supplement facts",
        "ingredients",
        "składniki",
        "skład",
        "zawartość",
        "nutrition facts",
    ]
    for kw in keywords:
        # Exact text search in common containers
        for tag in soup.find_all(["div", "section", "article", "p", "ul", "li"]):
            if kw in tag.get_text(separator=" ").lower():
                text = tag.get_text(separator=" ", strip=True)
                if 50 < len(text) < 5000:
                    return text

    return None


def _extract_name(soup: BeautifulSoup, url: str) -> str:
    """Extract product name from page."""
    # Structured data first
    el = soup.find(attrs={"itemprop": "name"})
    if el:
        name = el.get("content") or el.get_text(strip=True)
        if name:
            return name[:200]

    # Open Graph
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"][:200]

    # h1
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)[:200]

    # Title tag
    title = soup.find("title")
    if title:
        return title.get_text(strip=True)[:200]

    return url


def _extract_brand(soup: BeautifulSoup) -> Optional[str]:
    el = soup.find(attrs={"itemprop": "brand"})
    if el:
        name_el = el.find(attrs={"itemprop": "name"})
        if name_el:
            return name_el.get_text(strip=True)
        return el.get_text(strip=True)

    og = soup.find("meta", property="og:site_name")
    if og and og.get("content"):
        return og["content"]

    return None


async def scrape_product(url: str) -> ProductInfo:
    """Fetch product page and extract supplement information."""
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "lxml")

    # Remove scripts and styles for cleaner text extraction
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    name = _extract_name(soup, url)
    brand = _extract_brand(soup)
    price, currency = _extract_price(soup)
    ingredients_raw = _extract_ingredients(soup)

    # Fallback: try to get description for analysis when no ingredients found
    description = None
    if not ingredients_raw:
        og_desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", property="og:description"
        )
        if og_desc:
            description = og_desc.get("content", "")[:1000]

    return ProductInfo(
        url=url,
        name=name,
        brand=brand,
        price=price,
        currency=currency,
        ingredients_raw=ingredients_raw,
        description=description,
    )
