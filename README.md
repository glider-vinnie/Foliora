# Foliora — Library Book Demand Prediction

Full-stack starter: React + Vite + Tailwind-style CSS, FastAPI, Pandas, scikit-learn and XGBoost.

## Architecture
- `frontend/`: responsive library-themed dashboard and demand forecast UI
- `backend/`: FastAPI endpoints, data ingestion, feature engineering, model training and prediction
- `backend/data/`: downloaded dataset and generated model artifacts (created locally)

## Dataset
Seattle Public Library — Checkouts by Title: https://data.seattle.gov/Community/Checkouts-by-Title/tmmm-ytt6/about_data
CSV export: https://cos-data.seattle.gov/api/v3/views/tmmm-ytt6/export.csv?accessType=DOWNLOAD

The dataset can be large. Use `--limit-rows` to download a manageable sample for local development. The pipeline detects common column-name variants; check the source schema if the city's export changes.

## Run locally
### 1. Backend (Python 3.10+)
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py --limit-rows 250000
python scripts/train_model.py
uvicorn app.main:app --reload --port 8000
```

If the open data CSV is slow/unavailable, download it from the link above and save as `backend/data/checkouts.csv`, then run the scripts.

### 2. Frontend (Node 18+)
In another terminal:
```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (usually http://localhost:5173). The frontend calls `http://localhost:8000` by default. Set `VITE_API_URL` to override it.

## Model design
The target is monthly checkout count for a title. Features include lags, rolling means, month, and year. Models are compared on a chronological holdout split using MAE and RMSE. The API returns predictions from the saved trained model; it does not fabricate metric values. For demonstration, a small curated sample may have too little history for some titles, so use a title with multiple months of records.

## Important limitations
- Public-library data is not university-specific and may not include student/member or academic-calendar features.
- Checkout volume is not the same as copies needed; inventory recommendations require current stock, waitlists, and copy availability.
- This starter is intended for local development. Add caching, data refresh jobs, authentication and production hosting configuration before deployment.
