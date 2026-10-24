import argparse
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SODA_URL = "https://data.seattle.gov/resource/tmmm-ytt6.csv"
EXPORT_URL = "https://cos-data.seattle.gov/api/v3/views/tmmm-ytt6/export.csv?accessType=DOWNLOAD"

def main():
    parser = argparse.ArgumentParser(description="Download Seattle Public Library checkout data.")
    parser.add_argument("--limit-rows", type=int, default=50000, help="Maximum CSV rows to keep (default: 50,000).")
    parser.add_argument("--input", type=str, help="Use a previously downloaded CSV instead of downloading.")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "checkouts.csv"

    if args.input:
        print(f"Loading local CSV from {args.input}...")
        df = pd.read_csv(args.input, low_memory=False, nrows=args.limit_rows or None)
        df.to_csv(out, index=False)
    else:
        limit = args.limit_rows or 50000
        print(f"Fetching {limit:,} records from Seattle Open Data API...")
        dfs = []
        fetched = 0
        batch_size = min(limit, 50000)
        
        while fetched < limit:
            current_batch = min(batch_size, limit - fetched)
            params = {
                "$where": "title >= 'A'",
                "$order": "title,checkoutyear,checkoutmonth",
                "$limit": current_batch,
                "$offset": fetched,
            }
            try:
                resp = requests.get(SODA_URL, params=params, timeout=60)
                resp.raise_for_status()
                batch_df = pd.read_csv(pd.io.common.StringIO(resp.text), low_memory=False)
                if batch_df.empty:
                    break
                dfs.append(batch_df)
                fetched += len(batch_df)
                print(f"Downloaded {fetched:,}/{limit:,} rows...")
                if len(batch_df) < current_batch:
                    break
            except Exception as e:
                print(f"SODA API request failed ({e}). Falling back to streaming direct download...")
                with requests.get(EXPORT_URL, stream=True, timeout=(20, 180)) as response:
                    response.raise_for_status()
                    with out.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                if out.stat().st_size > 50 * 1024 * 1024:
                                    break
                df = pd.read_csv(out, low_memory=False, nrows=limit)
                df.to_csv(out, index=False)
                dfs = [df]
                break
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df.to_csv(out, index=False)

    print(f"Saved {len(df):,} rows to {out}")
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
