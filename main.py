from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

from scraper import scrape_product
from analyzer import analyze_supplement, generate_alternatives

app = FastAPI(title="Supplement Price Finder", version="1.0.0")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    product: dict
    analysis: dict
    alternatives: dict


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    url = request.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Nieprawidłowy URL. Podaj pełny adres zaczynający się od http:// lub https://")

    # 1. Scrape product page
    log.info(f"[1/3] Pobieram stronę produktu: {url}")
    try:
        product = await scrape_product(url)
        log.info(f"[1/3] OK – produkt: {product.name}")
    except Exception as e:
        log.error(f"[1/3] BŁĄD scraping: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Nie udało się pobrać strony produktu: {str(e)}"
        )

    if not product.ingredients_raw and not product.description:
        raise HTTPException(
            status_code=422,
            detail="Nie znaleziono informacji o składnikach na podanej stronie. Upewnij się, że link prowadzi do strony konkretnego produktu suplementacyjnego."
        )

    # 2. Analyze with Claude
    log.info("[2/3] Analizuję skład z Claude AI...")
    try:
        analysis = analyze_supplement(product)
        log.info(f"[2/3] OK – kategoria: {analysis.category}, składniki: {len(analysis.ingredients)}")
    except Exception as e:
        log.error(f"[2/3] BŁĄD analizy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Błąd analizy składników: {str(e)}"
        )

    # 3. Generate alternatives
    log.info("[3/3] Generuję tańsze zamienniki z Claude AI...")
    try:
        alternatives = generate_alternatives(analysis, product.price)
        log.info("[3/3] OK – gotowe!")
    except Exception as e:
        log.error(f"[3/3] BŁĄD alternatyw: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Błąd generowania alternatyw: {str(e)}"
        )

    return AnalyzeResponse(
        product=product.model_dump(),
        analysis=analysis.model_dump(),
        alternatives=alternatives,
    )


@app.get("/api/health")
async def health():
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))
    return {"status": "ok", "api_key_configured": api_key_set}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
