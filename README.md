# HengSeng & KRX Sentiment Price Prediction Agent (n+1)

**CrowdWisdomTrading — AI Agents Developer Intern Assessment**

A Python-based multi-source social & news sentiment pipeline that scrapes 10 days of retail investor posts across **Hong Kong (HKEX / Hang Seng)** and **South Korea (KRX / KOSPI)** using **Apify**, complements social signals with neural news ingestion via **Exa.com**, aligns all records to UTC price bars, extracts support/resistance & trading positions, and forecasts asset prices at **$n+1$** using strict walk-forward out-of-sample validation.

---

## 1. Exchange Coverage & Social Sources Used (Min. 2 per Exchange)

### Hong Kong Exchange (HKEX — 5 Assets)
* **Tickers:** `0700.HK` (Tencent), `1810.HK` (Xiaomi), `1211.HK` (BYD), `3690.HK` (Meituan), `9988.HK` (Alibaba)
* **Social Sources Scraped via Apify:**
  1. **LIHKG (連登討論區 — 財經台 `/category/15`):** Scraped via `apify/puppeteer-scraper`
  2. **Futu / Moomoo Community (富途牛牛圈):** Scraped via `apify/web-scraper`
  3. **AAStocks & ETNet Discussion Boards / Telegram HK Channels:** Scraped via `apify/cheerio-scraper` & `apify/telegram-scraper`

### South Korea Exchange (KRX — 5 Assets)
* **Tickers:** `000660.KS` (SK Hynix), `005380.KS` (Hyundai Motor), `035420.KS` (NAVER), `005930.KS` (Samsung Electronics), `035720.KS` (Kakao)
* **Social Sources Scraped via Apify:**
  1. **Naver Finance Stock Discussion Boards (네이버 증권 종목토론실):** Scraped via `apify/cheerio-scraper`
  2. **Daum Finance Forums (다음 금융 종목토론):** Scraped via `apify/web-scraper`
  3. **Blind Korea (블라인드) & Telegram/Kakao Open Stock Channels:** Scraped via `apify/puppeteer-scraper` & `apify/telegram-scraper`

### News Sentiment Complement
* **Exa.com (`exa-py`):** Neural search across financial publications over the last 10 days for each ticker, cashtag, and localized keyword.

---

## 2. Apify Actor IDs & Reproducibility Configuration

| Source Platform | Apify Actor ID | Query Modes | Extracted Fields |
| :--- | :--- | :--- | :--- |
| **Naver 종목토론실 & AAStocks** | `apify/cheerio-scraper` | Ticker, Keyword | `timestamp_utc`, `text`, `likes`, `reposts`, `views`, `author_followers` |
| **Futu / Moomoo & Daum Finance** | `apify/web-scraper` | Cashtag (`$0700.HK`), Ticker | `timestamp_utc`, `text`, `likes`, `reposts`, `views`, `author_followers` |
| **LIHKG & Blind Korea** | `apify/puppeteer-scraper` | Ticker, Cashtag, Keyword | `timestamp_utc`, `text`, `likes`, `reposts`, `views`, `author_followers` |
| **Telegram HK & KRX Channels** | `apify/telegram-scraper` | Keyword, Cashtag | `timestamp_utc`, `text`, `views`, `forwards`, `channel_subscribers` |

---

## 3. Quickstart & Reproducing the Pipeline

```bash
# 1. Clone the repository and install dependencies
pip install -r requirements.txt

# 2. Configure your free API tokens
cp .env.example .env
# Edit .env with your APIFY_API_TOKEN, EXA_API_KEY, and optional TAVILY_API_KEY

# 3. Run the end-to-end ingestion, UTC bar alignment, n+1 prediction, and visual report generator
python main_pipeline.py
```

### Output Artifacts
* `outputs/ticker_predictions_n_plus_1.csv` — Full table sorted by social mentions, sentiment score, direction, position, S/R levels, and $n+1$ price forecast.
* `outputs/visual_report_hkex_krx.png` — Multi-panel visual report sorting tickers by social mentions, sentiment score, and $n+1$ predicted return per exchange.
