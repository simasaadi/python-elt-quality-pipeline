from __future__ import annotations

from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "processed" / "retail_star_schema.duckdb"
OUT_DIR = REPO_ROOT / "outputs" / "artifacts"


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")

    # 1) Store daily performance mart
    con.execute("DROP TABLE IF EXISTS analytics.mart_store_daily;")
    con.execute(
        """
        CREATE TABLE analytics.mart_store_daily AS
        SELECT
            CAST(f.sales_date AS DATE) AS sales_day,
            f.store_sk,
            s.store_name,
            COUNT(*) AS transactions,
            SUM(f.total_amount) AS revenue,
            AVG(f.total_amount) AS avg_order_value
        FROM mart.fct_sales f
        LEFT JOIN mart.dim_stores s
          ON f.store_sk = s.store_sk
        GROUP BY 1,2,3;
        """
    )

    # 2) Campaign performance mart
    con.execute("DROP TABLE IF EXISTS analytics.mart_campaign_perf;")
    con.execute(
        """
        CREATE TABLE analytics.mart_campaign_perf AS
        SELECT
            f.campaign_sk,
            c.campaign_name,
            COUNT(*) AS transactions,
            SUM(f.total_amount) AS revenue,
            AVG(f.total_amount) AS avg_order_value,
            MIN(CAST(f.sales_date AS DATE)) AS first_sale_day,
            MAX(CAST(f.sales_date AS DATE)) AS last_sale_day
        FROM mart.fct_sales f
        LEFT JOIN mart.dim_campaigns c
          ON f.campaign_sk = c.campaign_sk
        GROUP BY 1,2;
        """
    )

    # Export artifacts
    con.execute(f"COPY analytics.mart_store_daily TO '{(OUT_DIR / 'mart_store_daily.parquet').as_posix()}' (FORMAT PARQUET);")
    con.execute(f"COPY analytics.mart_store_daily TO '{(OUT_DIR / 'mart_store_daily.csv').as_posix()}' (HEADER, DELIMITER ',');")

    con.execute(f"COPY analytics.mart_campaign_perf TO '{(OUT_DIR / 'mart_campaign_perf.parquet').as_posix()}' (FORMAT PARQUET);")
    con.execute(f"COPY analytics.mart_campaign_perf TO '{(OUT_DIR / 'mart_campaign_perf.csv').as_posix()}' (HEADER, DELIMITER ',');")

    print("✅ Built analytics marts + exported artifacts")
    print(f"Artifacts: {OUT_DIR}")


if __name__ == "__main__":
    main()
