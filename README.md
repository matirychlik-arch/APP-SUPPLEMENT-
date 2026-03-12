# 💊 Supplement Price Finder

Aplikacja webowa, która analizuje suplementy diety i znajduje **tańsze zamienniki** lub możliwość kupna **osobnych składników** – pozwala zaoszczędzić nawet 70% ceny.

## Jak działa?

1. Użytkownik wkleja link do produktu suplementacyjnego (iHerb, Bodybuilding.com, Myprotein, Allegro, Amazon itp.)
2. Aplikacja scrape'uje stronę produktu (nazwa, cena, składniki)
3. **Claude AI** analizuje skład i identyfikuje kluczowe składniki aktywne
4. AI generuje:
   - Tańsze zamienniki całego produktu (inne marki)
   - Opcję DIY stack – kupno każdego składnika osobno
   - Polecane sklepy
   - Szacowane oszczędności i porady

## Instalacja

### Wymagania
- Python 3.11+
- Klucz API Anthropic

### Uruchomienie

```bash
# 1. Sklonuj repo i wejdź do katalogu
cd APP-SUPPLEMENT-

# 2. Utwórz wirtualne środowisko
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub: venv\Scripts\activate  # Windows

# 3. Zainstaluj zależności
pip install -r requirements.txt

# 4. Skonfiguruj klucz API
cp .env.example .env
# Edytuj .env i wstaw swój klucz ANTHROPIC_API_KEY

# 5. Uruchom aplikację
python main.py
```

Otwórz przeglądarkę: http://localhost:8000

## Struktura projektu

```
├── main.py          # FastAPI app + endpointy
├── scraper.py       # Web scraper stron produktów
├── analyzer.py      # Integracja z Claude AI
├── static/
│   └── index.html   # Frontend (Tailwind CSS)
├── requirements.txt
└── .env.example
```

## Obsługiwane sklepy

- iHerb
- Bodybuilding.com
- Myprotein
- Amazon
- Allegro
- Sklepy z metadanymi schema.org (większość sklepów e-commerce)

## Technologie

- **Backend**: FastAPI + Python
- **AI**: Claude Sonnet (Anthropic)
- **Scraping**: httpx + BeautifulSoup4
- **Frontend**: HTML + Tailwind CSS (vanilla JS)
