from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "checkouts.csv"
MODEL_PATH = BASE / "data" / "model.joblib"
META_PATH = BASE / "data" / "model_meta.json"

ALIASES = {
    "title": ["title", "book title", "checkout title", "item title"],
    "count": ["checkouts", "checkout count", "total checkouts", "count", "usage count"],
    "year": ["checkout year", "year", "year of checkout"],
    "month": ["checkout month", "month", "month of checkout"],
    "date": ["checkout date", "date", "month date", "checkout month date"],
    "creator": ["creator", "author"],
    "material": ["material type", "item type", "format"],
}

def _col(df, key):
    norm = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for alias in ALIASES[key]:
        cleaned = re.sub(r"[^a-z0-9]+", "", alias.lower())
        if cleaned in norm: return norm[cleaned]
    return None

def load_monthly():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python scripts/prepare_data.py --limit-rows 250000")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    title_col, count_col = _col(raw, "title"), _col(raw, "count")
    if not title_col or not count_col:
        raise ValueError(f"Could not identify title and checkout-count columns. Found: {list(raw.columns)}")
    year_col, month_col, date_col = _col(raw, "year"), _col(raw, "month"), _col(raw, "date")
    if year_col and month_col:
        dates = pd.to_datetime(dict(year=pd.to_numeric(raw[year_col], errors="coerce"), month=pd.to_numeric(raw[month_col], errors="coerce"), day=1), errors="coerce")
    elif date_col:
        dates = pd.to_datetime(raw[date_col], errors="coerce")
    else:
        raise ValueError(f"Could not identify checkout year/month or date columns. Found: {list(raw.columns)}")
    df = pd.DataFrame({"title": raw[title_col].astype(str).str.strip(), "date": dates, "count": pd.to_numeric(raw[count_col], errors="coerce")})
    df = df.dropna(subset=["date", "count"])
    df = df[(df.title != "") & (df.title.str.lower() != "nan") & (df["count"] >= 0)]
    df["month"] = df.date.dt.to_period("M").dt.to_timestamp()
    # The source may already have one row per title/month; summing duplicate title-month rows is safe.
    monthly = df.groupby(["title", "month"], as_index=False)["count"].sum().rename(columns={"count":"demand"})
    return monthly

def features_for(group):
    g = group.sort_values("month").copy()
    g["month_num"] = g.month.dt.month
    g["year_num"] = g.month.dt.year
    g["lag_1"] = g.demand.shift(1)
    g["lag_2"] = g.demand.shift(2)
    g["lag_3"] = g.demand.shift(3)
    g["rolling_3"] = g.demand.shift(1).rolling(3, min_periods=3).mean()
    return g.dropna(subset=["lag_1", "lag_2", "lag_3", "rolling_3"])

FEATURES = ["month_num", "year_num", "lag_1", "lag_2", "lag_3", "rolling_3"]

def train():
    monthly = load_monthly()
    rows = []
    for title, group in monthly.groupby("title"):
        feat = features_for(group)
        if len(feat): rows.append(feat)
    if not rows: raise ValueError("Not enough monthly history to train. Download more data or choose a larger sample.")
    dataset = pd.concat(rows, ignore_index=True).sort_values("month")
    unique_months = sorted(dataset.month.unique())
    if len(unique_months) < 3: raise ValueError("Need at least 3 distinct months after feature engineering.")
    cutoff = unique_months[max(1, int(len(unique_months) * 0.8)) - 1]
    train_df, test_df = dataset[dataset.month <= cutoff], dataset[dataset.month > cutoff]
    if test_df.empty:
        cutoff = unique_months[-2]
        train_df, test_df = dataset[dataset.month <= cutoff], dataset[dataset.month > cutoff]
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=180, max_depth=6, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42, n_jobs=4),
    }
    metrics, fitted = [], {}
    X_train, y_train = train_df[FEATURES], train_df.demand
    X_test, y_test = test_df[FEATURES], test_df.demand
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.maximum(0, model.predict(X_test))
        metrics.append({"model":name, "mae":round(float(mean_absolute_error(y_test, pred)), 3), "rmse":round(float(np.sqrt(mean_squared_error(y_test, pred))), 3)})
        fitted[name] = model
    best = min(metrics, key=lambda m: m["mae"])["model"]
    # Refit chosen model on all available rows for serving future predictions.
    fitted[best].fit(dataset[FEATURES], dataset.demand)
    joblib.dump({"model":fitted[best], "features":FEATURES, "model_name":best}, MODEL_PATH)
    meta = {"model_name":best, "metrics":metrics, "rows":int(len(monthly)), "titles":int(monthly.title.nunique()), "start":str(monthly.month.min().date()), "end":str(monthly.month.max().date()), "trained_rows":int(len(dataset)), "holdout_start":str(pd.Timestamp(test_df.month.min()).date())}
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta

def metadata():
    if not META_PATH.exists(): raise FileNotFoundError("Model not trained yet. Run python scripts/train_model.py")
    return json.loads(META_PATH.read_text())

def load_bundle():
    if not MODEL_PATH.exists(): raise FileNotFoundError("Model not trained yet. Run python scripts/train_model.py")
    return joblib.load(MODEL_PATH)

def forecast(title: str, periods: int = 1):
    monthly = load_monthly()
    matches = monthly[monthly.title.str.casefold() == title.casefold()]
    if matches.empty:
        # fuzzy-ish substring search; exact selected title is preferred
        matches = monthly[monthly.title.str.casefold().str.contains(re.escape(title.casefold()), na=False)]
    if matches.empty: raise LookupError("No matching title found in the loaded dataset.")
    chosen = matches.groupby("title").demand.sum().sort_values(ascending=False).index[0]
    history = matches[matches.title == chosen].sort_values("month")
    if len(history) < 3: raise ValueError("This title has fewer than 3 monthly observations; choose a title with more history.")
    bundle = load_bundle(); model = bundle["model"]
    work = history[["month", "demand"]].copy()
    output = []
    for _ in range(max(1, min(periods, 12))):
        last_month = work.month.max()
        next_month = last_month + pd.offsets.MonthBegin(1)
        vals = work.demand.tail(3).tolist()
        row = pd.DataFrame([{"month_num":next_month.month, "year_num":next_month.year, "lag_1":vals[-1], "lag_2":vals[-2], "lag_3":vals[-3], "rolling_3":float(np.mean(vals))}])[FEATURES]
        pred = max(0, float(model.predict(row)[0]))
        output.append({"month":next_month.strftime("%Y-%m"), "predicted_demand":round(pred), "predicted_exact":round(pred, 2)})
        work = pd.concat([work, pd.DataFrame([{"month":next_month, "demand":pred}])], ignore_index=True)
    hist = [{"month":r.month.strftime("%Y-%m"), "demand":round(float(r.demand))} for r in history.tail(18).itertuples()]
    return {"title":chosen, "history":hist, "forecast":output, "model":bundle["model_name"]}
