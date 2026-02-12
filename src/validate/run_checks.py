from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "processed" / "retail_star_schema.duckdb"
OUT_PATH = REPO_ROOT / "outputs" / "reports" / "dq_results.json"


@dataclass
class CheckResult:
    check_name: str
    status: str  # PASS / FAIL
    failed_rows: int
    threshold_failed_rows: int
    details: str


def _run_count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0])


def _table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    return (
        _run_count(
            con,
            f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = '{schema}'
              AND table_name = '{table}';
            """,
        )
        == 1
    )


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found. Run ingest/transform first: {DB_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    # Make sure mart exists
    con.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name='mart';")

    results: list[CheckResult] = []

    def add(check_name: str, failed_rows: int, threshold: int, details: str) -> None:
        status = "PASS" if failed_rows <= threshold else "FAIL"
        results.append(CheckResult(check_name, status, failed_rows, threshold, details))

    # -------------------------
    # Row count floors (basic sanity)
    # -------------------------
    add(
        "mart.dim_customers__rowcount_floor",
        failed_rows=0 if _run_count(con, "SELECT COUNT(*) FROM mart.dim_customers") >= 1 else 1,
        threshold=0,
        details="dim_customers should have at least 1 row",
    )
    add(
        "mart.fct_sales__rowcount_floor",
        failed_rows=0 if _run_count(con, "SELECT COUNT(*) FROM mart.fct_sales") >= 1 else 1,
        threshold=0,
        details="fct_sales should have at least 1 row",
    )

    # -------------------------
    # NOT NULL checks (real fact columns)
    # -------------------------
    not_null_checks = [
        ("fct_sales", "sales_id"),
        ("fct_sales", "sales_date"),
        ("fct_sales", "customer_sk"),
        ("fct_sales", "product_sk"),
        ("fct_sales", "store_sk"),
        ("fct_sales", "salesperson_sk"),
        ("fct_sales", "total_amount"),
    ]

    for table, col in not_null_checks:
        failed = _run_count(con, f"SELECT COUNT(*) FROM mart.{table} WHERE {col} IS NULL;")
        add(
            f"mart.{table}__{col}__not_null",
            failed_rows=failed,
            threshold=0,
            details=f"{col} should never be NULL",
        )

    # -------------------------
    # Uniqueness checks (fact natural key)
    # -------------------------
    dupes_sales_id = _run_count(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT sales_id
            FROM mart.fct_sales
            GROUP BY sales_id
            HAVING COUNT(*) > 1
        ) t;
        """,
    )
    add(
        "mart.fct_sales__sales_id__unique",
        failed_rows=dupes_sales_id,
        threshold=0,
        details="sales_id should be unique in fct_sales",
    )

    # -------------------------
    # Referential integrity checks (SKs should resolve to dims)
    # -------------------------
    fk_checks = [
        ("customer_sk", "dim_customers", "customer_sk"),
        ("product_sk", "dim_products", "product_sk"),
        ("store_sk", "dim_stores", "store_sk"),
        ("salesperson_sk", "dim_salespersons", "salesperson_sk"),
        ("campaign_sk", "dim_campaigns", "campaign_sk"),
    ]

    for fct_col, dim_table, dim_col in fk_checks:
        orphans = _run_count(
            con,
            f"""
            SELECT COUNT(*)
            FROM mart.fct_sales f
            LEFT JOIN mart.{dim_table} d
              ON f.{fct_col} = d.{dim_col}
            WHERE f.{fct_col} IS NOT NULL
              AND d.{dim_col} IS NULL;
            """,
        )
        add(
            f"mart.fct_sales__{fct_col}_fk__references_{dim_table}.{dim_col}",
            failed_rows=orphans,
            threshold=0,
            details=f"All {fct_col} values in fct_sales should exist in {dim_table}.{dim_col}",
        )

    # -------------------------
    # Business rule checks
    # -------------------------
    neg_amount = _run_count(con, "SELECT COUNT(*) FROM mart.fct_sales WHERE total_amount < 0;")
    add(
        "mart.fct_sales__total_amount__non_negative",
        failed_rows=neg_amount,
        threshold=0,
        details="total_amount should be >= 0",
    )

    future_dates = _run_count(con, "SELECT COUNT(*) FROM mart.fct_sales WHERE sales_date > now() + INTERVAL 1 day;")
    add(
        "mart.fct_sales__sales_date__not_future",
        failed_rows=future_dates,
        threshold=0,
        details="sales_date should not be in the future (beyond 1 day)",
    )

    huge_amount = _run_count(con, "SELECT COUNT(*) FROM mart.fct_sales WHERE total_amount > 1000000;")
    add(
        "mart.fct_sales__total_amount__not_absurd",
        failed_rows=huge_amount,
        threshold=0,
        details="total_amount should not exceed 1,000,000 (sanity check)",
    )

    # -------------------------
    # Analytics mart checks (only if the tables exist)
    # -------------------------
    if _table_exists(con, "analytics", "mart_store_daily"):
        store_daily_rows = _run_count(con, "SELECT COUNT(*) FROM analytics.mart_store_daily;")
        add(
            "analytics.mart_store_daily__rowcount_floor",
            failed_rows=0 if store_daily_rows >= 1 else 1,
            threshold=0,
            details="mart_store_daily should have at least 1 row",
        )

        neg_rev = _run_count(con, "SELECT COUNT(*) FROM analytics.mart_store_daily WHERE revenue < 0;")
        add(
            "analytics.mart_store_daily__revenue__non_negative",
            failed_rows=neg_rev,
            threshold=0,
            details="revenue should be >= 0",
        )

    if _table_exists(con, "analytics", "mart_campaign_perf"):
        camp_null_name = _run_count(
            con, "SELECT COUNT(*) FROM analytics.mart_campaign_perf WHERE campaign_name IS NULL;"
        )
        add(
            "analytics.mart_campaign_perf__campaign_name__not_null",
            failed_rows=camp_null_name,
            threshold=0,
            details="campaign_name should not be NULL (dim join should resolve)",
        )

    payload = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path": str(DB_PATH),
        "results": [r.__dict__ for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == "PASS"),
            "failed": sum(1 for r in results if r.status == "FAIL"),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("✅ Data quality checks complete")
    print(f"Report: {OUT_PATH}")
    print(f"Passed: {payload['summary']['passed']} / {payload['summary']['total']}")


if __name__ == "__main__":
    main()
