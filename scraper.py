import json
import logging
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import Optional
import re

try:
    from curl_cffi.requests import AsyncSession as CurlSession
    _CURL_AVAILABLE = True
except ImportError:
    _CURL_AVAILABLE = False

log = logging.getLogger(__name__)


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
    rating: Optional[str] = None
    review_count: Optional[str] = None


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _detect_currency(text: str) -> str:
    if "$" in text or "USD" in text:
        return "USD"
    if "€" in text or "EUR" in text:
        return "EUR"
    if "£" in text or "GBP" in text:
        return "GBP"
    return "PLN"


def _extract_rating_from_json_ld(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """Extract aggregateRating from JSON-LD structured data."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            agg = item.get("aggregateRating")
            if agg:
                rating = str(agg.get("ratingValue", "") or "").strip() or None
                count = str(agg.get("reviewCount", "") or agg.get("ratingCount", "") or "").strip() or None
                if rating:
                    return rating, count
    return None, None


def _extract_rating(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """Try JSON-LD first, then itemprop, then CSS selectors."""
    rating, count = _extract_rating_from_json_ld(soup)
    if rating:
        return rating, count

    # itemprop
    r_el = soup.find(attrs={"itemprop": "ratingValue"})
    c_el = soup.find(attrs={"itemprop": "reviewCount"}) or soup.find(attrs={"itemprop": "ratingCount"})
    if r_el:
        rating = r_el.get("content") or r_el.get_text(strip=True)
        count = (c_el.get("content") or c_el.get_text(strip=True)) if c_el else None
        return rating or None, count or None

    # CSS selectors
    for sel in ["[data-rating]", ".rating-value", ".average-rating", ".star-rating"]:
        el = soup.select_one(sel)
        if el:
            val = el.get("data-rating") or el.get_text(strip=True)
            match = re.search(r"\d+\.?\d*", val)
            if match:
                return match.group(), None

    return None, None


def _extract_price_from_json_ld(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """Extract price from JSON-LD structured data (Shopify, schema.org)."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue

        # Handle list of objects
        items = data if isinstance(data, list) else [data]
        for item in items:
            # Direct Product offer
            offers = item.get("offers") or item.get("Offers")
            if offers:
                if isinstance(offers, list):
                    offers = offers[0]
                price = offers.get("price") or offers.get("lowPrice")
                currency = offers.get("priceCurrency", "PLN")
                if price:
                    return str(price), currency
            # AggregateOffer
            price = item.get("price")
            currency = item.get("priceCurrency", "PLN")
            if price:
                return str(price), currency

    return None, None


def _extract_price(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """Try JSON-LD first, then common CSS selectors."""
    # 1. JSON-LD (Shopify, WooCommerce, etc.)
    price, currency = _extract_price_from_json_ld(soup)
    if price:
        return price, currency

    # 2. HTML selectors
    price_selectors = [
        'span[itemprop="price"]',
        'meta[itemprop="price"]',
        ".price",
        ".product-price",
        ".current-price",
        ".sale-price",
        '[class*="price"]',
        ".cena",
        ".price-final",
    ]
    for sel in price_selectors:
        el = soup.select_one(sel)
        if el:
            content = el.get("content") or el.get_text(strip=True)
            match = re.search(r"\d[\d\s]*[,.]?\d*", content)
            if match:
                raw = match.group().replace(" ", "").replace(",", ".")
                return raw, _detect_currency(el.get_text(" ", strip=True))

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


async def _fetch_html(url: str) -> str:
    """Fetch page HTML, using curl_cffi (Chrome impersonation) first, httpx as fallback."""
    if _CURL_AVAILABLE:
        try:
            async with CurlSession(impersonate="chrome120") as session:
                resp = await session.get(url, timeout=25, allow_redirects=True)
                if resp.status_code < 400:
                    log.info(f"curl_cffi OK ({resp.status_code})")
                    return resp.text
                log.warning(f"curl_cffi got {resp.status_code}, falling back to httpx")
        except Exception as e:
            log.warning(f"curl_cffi failed ({e}), falling back to httpx")

    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def scrape_product(url: str) -> ProductInfo:
    """Fetch product page and extract supplement information."""
    html = await _fetch_html(url)

    soup = BeautifulSoup(html, "lxml")

    # Extract price and rating BEFORE removing scripts (JSON-LD is inside <script> tags)
    name = _extract_name(soup, url)
    brand = _extract_brand(soup)
    price, currency = _extract_price(soup)
    rating, review_count = _extract_rating(soup)

    # Now remove scripts and styles for cleaner text extraction
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
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
        rating=rating,
        review_count=review_count,
    )
