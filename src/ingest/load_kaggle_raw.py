from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "retail-star-schema-elt"
DB_PATH = REPO_ROOT / "data" / "processed" / "retail_star_schema.duckdb"
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "ingest_summary.json"

CSV_FILES = [
    "dim_campaigns.csv",
    "dim_customers.csv",
    "dim_dates.csv",
    "dim_products.csv",
    "dim_salespersons.csv",
    "dim_stores.csv",
    "fact_sales_denormalized.csv",
    "fact_sales_normalized.csv",
]


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {RAW_DIR}")

    missing = [f for f in CSV_FILES if not (RAW_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected CSV files in raw folder:\n"
            + "\n".join(str(RAW_DIR / f) for f in missing)
        )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA enable_progress_bar=false;")

    # Keep it explicit and consistent: raw.<table_name>
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

    summary: dict[str, object] = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": str(RAW_DIR),
        "db_path": str(DB_PATH),
        "tables": [],
    }

    for fname in CSV_FILES:
        fpath = RAW_DIR / fname
        table = fname.replace(".csv", "")
        full_table = f"raw.{table}"

        # Rebuild raw layer on each ingest (simple + reliable for portfolio)
        con.execute(f"DROP TABLE IF EXISTS {full_table};")
        con.execute(
            f"""
            CREATE TABLE {full_table} AS
            SELECT *
            FROM read_csv_auto('{fpath.as_posix()}', header=true);
            """
        )

        row_count = con.execute(f"SELECT COUNT(*) FROM {full_table};").fetchone()[0]
        cols = con.execute(f"PRAGMA table_info('{full_table}');").fetchall()
        col_names = [c[1] for c in cols]

        summary["tables"].append(
            {
                "file": fname,
                "table": full_table,
                "rows": int(row_count),
                "columns": len(col_names),
            }
        )

    # Write a small artifact we can display in report/CI later
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"✅ Loaded {len(CSV_FILES)} CSVs into DuckDB raw schema")
    print(f"DB: {DB_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
