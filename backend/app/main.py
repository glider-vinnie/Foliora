from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .ml import load_monthly, metadata, forecast, train, DATA_PATH

app = FastAPI(title="Foliora API", version="1.0.0", description="Foliora library checkout demand forecasting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class ForecastRequest(BaseModel):
    title: str = Field(min_length=2, max_length=300)
    periods: int = Field(default=3, ge=1, le=12)

@app.get("/api/health")
def health(): return {"status":"ok", "dataset_ready":DATA_PATH.exists()}

@app.get("/api/dashboard")
def dashboard():
    try:
        df = load_monthly()
        month = df.groupby("month", as_index=False).demand.sum().sort_values("month")
        titles = df.groupby("title").demand.sum().sort_values(ascending=False)
        return {"total_checkouts":int(df.demand.sum()), "titles":int(df.title.nunique()), "monthly_records":int(len(df)), "date_range":{"start":str(df.month.min().date()),"end":str(df.month.max().date())}, "monthly_trend":[{"month":r.month.strftime("%Y-%m"),"checkouts":int(r.demand)} for r in month.tail(18).itertuples()], "top_titles":[{"title":str(k),"checkouts":int(v)} for k,v in titles.head(8).items()], "model":metadata() if (DATA_PATH.parent / "model_meta.json").exists() else None}
    except Exception as e: raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/books")
def books(q: str = Query(default="", max_length=100), limit: int = Query(default=50, ge=1, le=100)):
    try:
        df = load_monthly()
        summary = df.groupby("title").agg(total_checkouts=("demand","sum"), months=("month","nunique"), latest_month=("month","max")).reset_index()
        if q: summary = summary[summary.title.str.contains(q, case=False, na=False)]
        summary = summary.sort_values("total_checkouts", ascending=False).head(limit)
        return {"books":[{"title":r.title,"total_checkouts":int(r.total_checkouts),"months":int(r.months),"latest_month":r.latest_month.strftime("%Y-%m")} for r in summary.itertuples()]}
    except Exception as e: raise HTTPException(status_code=503, detail=str(e))

@app.post("/api/predict")
def predict(req: ForecastRequest):
    try: return forecast(req.title, req.periods)
    except LookupError as e: raise HTTPException(status_code=404, detail=str(e))
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/models")
def models():
    try: return metadata()
    except Exception as e: raise HTTPException(status_code=503, detail=str(e))

@app.post("/api/train")
def retrain():
    try: return train()
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))
