from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import Optional
import secrets
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

from scraper import scrape_product
from analyzer import analyze_supplement, generate_alternatives, analyze_quality

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rate limiter – max 10 zapytań/minutę per IP
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Supplement Price Finder", version="1.0.0", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS – tylko własna domena (+ localhost do developmentu)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


class AnalyzeRequest(BaseModel):
    url: str
    user_profile: Optional[str] = None


class AnalyzeResponse(BaseModel):
    product: dict
    analysis: dict
    alternatives: dict
    quality: dict


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")

def verify_token(x_access_token: str = Header(default="")):
    if ACCESS_TOKEN and not secrets.compare_digest(x_access_token, ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Nieprawidłowy token dostępu.")


@app.post("/api/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze(request: Request, body: AnalyzeRequest,
                  x_access_token: str = Header(default="")):
    if ACCESS_TOKEN and not secrets.compare_digest(x_access_token, ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Nieprawidłowy token dostępu.")

    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Nieprawidłowy URL. Podaj pełny adres zaczynający się od http:// lub https://")
    if len(url) > 2048:
        raise HTTPException(status_code=400, detail="URL jest zbyt długi.")

    # 1. Scrape product page
    log.info(f"[1/4] Pobieram stronę produktu: {url}")
    try:
        product = await scrape_product(url)
        log.info(f"[1/4] OK – produkt: {product.name}, rating: {product.rating}")
    except Exception as e:
        log.error(f"[1/4] BŁĄD scraping: {e}")
        raise HTTPException(status_code=422, detail=f"Nie udało się pobrać strony produktu: {str(e)}")

    if not product.ingredients_raw and not product.description:
        raise HTTPException(status_code=422, detail="Nie znaleziono informacji o składnikach na podanej stronie.")

    # 2. Analyze with Claude
    log.info("[2/4] Analizuję skład z Claude AI...")
    try:
        analysis = analyze_supplement(product)
        log.info(f"[2/4] OK – kategoria: {analysis.category}, składniki: {len(analysis.ingredients)}")
    except Exception as e:
        log.error(f"[2/4] BŁĄD analizy: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd analizy składników: {str(e)}")

    # 3. Generate alternatives
    log.info("[3/4] Generuję tańsze zamienniki z Claude AI...")
    try:
        alternatives = generate_alternatives(analysis, product.price, body.user_profile)
        log.info("[3/4] OK – gotowe!")
    except Exception as e:
        log.error(f"[3/4] BŁĄD alternatyw: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd generowania alternatyw: {str(e)}")

    # 4. Analyze ingredient quality + SupScore
    log.info("[4/4] Analizuję jakość składników i liczę SupScore...")
    quality = analyze_quality(
        analysis,
        original_price=product.price,
        rating=product.rating,
        review_count=product.review_count,
        user_profile=body.user_profile,
    )
    log.info(f"[4/4] OK – SupScore={quality.get('sup_score')}")

    return AnalyzeResponse(
        product=product.model_dump(),
        analysis=analysis.model_dump(),
        alternatives=alternatives,
        quality=quality,
    )


@app.get("/api/health")
async def health(x_access_token: str = Header(default="")):
    if ACCESS_TOKEN and not secrets.compare_digest(x_access_token, ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Nieprawidłowy token dostępu.")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
