from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "processed" / "retail_star_schema.duckdb"
OUT_DIR = REPO_ROOT / "outputs" / "artifacts"


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB not found. Run ingest first: {DB_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS mart;")

    # 1) Curated dims (simple pass-through, but in mart schema)
    dims = [
        "dim_campaigns",
        "dim_customers",
        "dim_dates",
        "dim_products",
        "dim_salespersons",
        "dim_stores",
    ]

    for t in dims:
        con.execute(f"DROP TABLE IF EXISTS mart.{t};")
        con.execute(f"CREATE TABLE mart.{t} AS SELECT * FROM raw.{t};")

    # 2) Curated fact table: pick the normalized one as the canonical fact
    con.execute("DROP TABLE IF EXISTS mart.fct_sales;")
    con.execute(
        """
        CREATE TABLE mart.fct_sales AS
        SELECT *
        FROM raw.fact_sales_normalized;
        """
    )

    # 3) Basic run metadata table (nice portfolio touch)
    con.execute("CREATE TABLE IF NOT EXISTS mart.pipeline_runs(run_utc VARCHAR, note VARCHAR);")
    con.execute(
        "INSERT INTO mart.pipeline_runs VALUES (?, ?);",
        [datetime.now(timezone.utc).isoformat(timespec="seconds"), "build_marts"],
    )

    # 4) Export artifacts (Parquet + CSV)
    exports = dims + ["fct_sales"]

    for t in exports:
        parquet_path = (OUT_DIR / f"{t}.parquet").as_posix()
        csv_path = (OUT_DIR / f"{t}.csv").as_posix()

        con.execute(f"COPY mart.{t} TO '{parquet_path}' (FORMAT PARQUET);")
        con.execute(f"COPY mart.{t} TO '{csv_path}' (HEADER, DELIMITER ',');")

    print("✅ Built mart schema + exported artifacts")
    print(f"DB: {DB_PATH}")
    print(f"Artifacts: {OUT_DIR}")


if __name__ == "__main__":
    main()
