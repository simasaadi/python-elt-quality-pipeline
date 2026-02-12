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
    # Null checks (keys should not be null)
    # NOTE: We don't assume exact column names; we probe for likely key columns.
    # -------------------------
    # Helper: only run check if column exists
    def col_exists(table: str, col: str) -> bool:
        q = f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema='mart'
          AND table_name='{table}'
          AND column_name='{col}';
        """
        return _run_count(con, q) == 1

    # Candidate key columns (common in star schemas)
    key_candidates = {
        "dim_customers": ["customer_id", "CustomerID", "customer_key"],
        "dim_products": ["product_id", "ProductID", "product_key"],
        "dim_stores": ["store_id", "StoreID", "store_key"],
        "dim_salespersons": ["salesperson_id", "SalespersonID", "salesperson_key"],
        "dim_dates": ["date_id", "DateID", "date_key", "date"],
        "dim_campaigns": ["campaign_id", "CampaignID", "campaign_key"],
        "fct_sales": ["sale_id", "SalesID", "sales_id", "transaction_id"],
    }

    for table, cols in key_candidates.items():
        for col in cols:
            if col_exists(table, col):
                failed = _run_count(con, f"SELECT COUNT(*) FROM mart.{table} WHERE {col} IS NULL;")
                add(
                    f"mart.{table}__{col}__not_null",
                    failed_rows=failed,
                    threshold=0,
                    details=f"{col} should never be NULL",
                )
                break  # only check the first matching key column

    # -------------------------
    # Uniqueness checks on dim keys (first matching key column)
    # -------------------------
    dim_tables = ["dim_customers", "dim_products", "dim_stores", "dim_salespersons", "dim_dates", "dim_campaigns"]

    for table in dim_tables:
        cols = key_candidates.get(table, [])
        for col in cols:
            if col_exists(table, col):
                dupes = _run_count(
                    con,
                    f"""
                    SELECT COUNT(*)
                    FROM (
                        SELECT {col}
                        FROM mart.{table}
                        GROUP BY {col}
                        HAVING COUNT(*) > 1
                    ) t;
                    """,
                )
                add(
                    f"mart.{table}__{col}__unique",
                    failed_rows=dupes,
                    threshold=0,
                    details=f"{col} should be unique in {table}",
                )
                break

    # -------------------------
    # Referential integrity checks (only if the columns exist)
    # We check that fct_sales keys exist in corresponding dims.
    # -------------------------
    fk_pairs = [
        ("fct_sales", ["customer_id", "CustomerID"], "dim_customers", ["customer_id", "CustomerID"]),
        ("fct_sales", ["product_id", "ProductID"], "dim_products", ["product_id", "ProductID"]),
        ("fct_sales", ["store_id", "StoreID"], "dim_stores", ["store_id", "StoreID"]),
        ("fct_sales", ["salesperson_id", "SalespersonID"], "dim_salespersons", ["salesperson_id", "SalespersonID"]),
    ]

    for fct, fct_cols, dim, dim_cols in fk_pairs:
        fct_col = next((c for c in fct_cols if col_exists(fct, c)), None)
        dim_col = next((c for c in dim_cols if col_exists(dim, c)), None)
        if fct_col and dim_col:
            orphans = _run_count(
                con,
                f"""
                SELECT COUNT(*)
                FROM mart.{fct} f
                LEFT JOIN mart.{dim} d
                  ON f.{fct_col} = d.{dim_col}
                WHERE f.{fct_col} IS NOT NULL
                  AND d.{dim_col} IS NULL;
                """,
            )
            add(
                f"mart.{fct}__{fct_col}_fk__references_{dim}.{dim_col}",
                failed_rows=orphans,
                threshold=0,
                details=f"All {fct_col} values in {fct} should exist in {dim}.{dim_col}",
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
