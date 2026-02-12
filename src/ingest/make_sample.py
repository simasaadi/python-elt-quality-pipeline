from __future__ import annotations

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "retail-star-schema-elt"
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "retail-star-schema-elt"

FILES = [
    "dim_campaigns.csv",
    "dim_customers.csv",
    "dim_dates.csv",
    "dim_products.csv",
    "dim_salespersons.csv",
    "dim_stores.csv",
    "fact_sales_denormalized.csv",
    "fact_sales_normalized.csv",
]

# keep dims full (they're small), sample facts to keep repo light
FACT_SAMPLE_N = 20000

def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for f in FILES:
        src = RAW_DIR / f
        dst = SAMPLE_DIR / f

        df = pd.read_csv(src)

        if f.startswith("fact_"):
            df = df.sample(n=min(FACT_SAMPLE_N, len(df)), random_state=42)

        df.to_csv(dst, index=False)

    print(f"✅ Wrote sample files to: {SAMPLE_DIR}")

if __name__ == "__main__":
    main()
