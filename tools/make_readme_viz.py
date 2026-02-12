from __future__ import annotations

from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "processed" / "retail_star_schema.duckdb"

DOCS_DIR = REPO_ROOT / "docs"
OUT_HTML = DOCS_DIR / "dq_dashboard.html"
OUT_PNG = DOCS_DIR / "dq_scorecard.png"


def ensure_docs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def list_tables(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    return con.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
          AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name
        """
    ).fetchall()


def describe_columns(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE {schema}.{table}").fetchall()]


def compute_checks_fast(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Fast checks only:
      - NULL counts for *_id / *_sk columns (cheap)
      - FK missing checks for mart.fct_sales (only when dim contains same column)
    Skips duplicates because they can be slow on large tables.
    """
    run_ts = datetime.now()

    all_tables = list_tables(con)
    mart_tables = [(s, t) for (s, t) in all_tables if s == "mart"]
    if not mart_tables:
        raise SystemExit("No mart tables found in the DB.")

    checks: list[dict] = []

    print(f"[1/3] Found {len(mart_tables)} mart tables. Running NULL checks...")

    # NULL checks
    for schema, table in mart_tables:
        cols = describe_columns(con, schema, table)
        keyish = [c for c in cols if c.lower().endswith("_id") or c.lower().endswith("_sk")]
        for col in keyish[:12]:
            nulls = con.execute(
                f"SELECT COUNT(*) FROM {schema}.{table} WHERE {col} IS NULL"
            ).fetchone()[0]
            checks.append(
                {
                    "run_ts": run_ts,
                    "check_type": "nulls",
                    "table": f"{schema}.{table}",
                    "column": col,
                    "dq_check_name": f"{schema}.{table}__{col}__nulls",
                    "status": "PASS" if int(nulls) == 0 else "FAIL",
                    "failed_rows": int(nulls),
                }
            )

    print(f"[2/3] NULL checks done ({len(checks)} checks). Running FK checks from mart.fct_sales...")

    # FK checks
    mart_table_names = {t for (s, t) in mart_tables}
    if "fct_sales" in mart_table_names:
        fact_schema, fact_table = "mart", "fct_sales"
        fact_cols = describe_columns(con, fact_schema, fact_table)
        fact_fks = [c for c in fact_cols if c.lower().endswith("_id") or c.lower().endswith("_sk")]

        dim_tables = [(s, t) for (s, t) in mart_tables if t.startswith("dim_")]

        def base_token(col: str) -> str:
            return col.lower().replace("_id", "").replace("_sk", "")

        for fk in fact_fks[:20]:
            fk_base = base_token(fk)

            best: tuple[str, str] | None = None

            # prefer dims whose name includes the fk token AND contain the fk column
            for ds, dt in dim_tables:
                if fk_base in dt.lower():
                    dim_cols = describe_columns(con, ds, dt)
                    if fk in dim_cols:
                        best = (ds, dt)
                        break

            # otherwise, any dim that contains fk column
            if best is None:
                for ds, dt in dim_tables:
                    dim_cols = describe_columns(con, ds, dt)
                    if fk in dim_cols:
                        best = (ds, dt)
                        break

            if best is None:
                continue

            ds, dt = best
            missing = con.execute(
                f"""
                SELECT COUNT(*)
                FROM {fact_schema}.{fact_table} f
                LEFT JOIN {ds}.{dt} d
                  ON f.{fk} = d.{fk}
                WHERE f.{fk} IS NOT NULL
                  AND d.{fk} IS NULL
                """
            ).fetchone()[0]

            checks.append(
                {
                    "run_ts": run_ts,
                    "check_type": "fk_missing",
                    "table": f"{fact_schema}.{fact_table}",
                    "column": fk,
                    "dq_check_name": f"{fact_schema}.{fact_table}__{fk}__missing_in_{ds}.{dt}",
                    "status": "PASS" if int(missing) == 0 else "FAIL",
                    "failed_rows": int(missing),
                }
            )

    print(f"[3/3] FK checks done. Total checks: {len(checks)}")
    df = pd.DataFrame(checks)
    if df.empty:
        raise SystemExit("No checks produced.")
    return df


def build_dashboard(df: pd.DataFrame) -> None:
    df = df.copy()
    df["run_ts"] = pd.to_datetime(df["run_ts"])
    df["status"] = df["status"].astype(str).str.upper()
    df["failed_rows"] = pd.to_numeric(df["failed_rows"], errors="coerce").fillna(0).astype(int)

    latest_ts = df["run_ts"].max()
    latest = df[df["run_ts"] == latest_ts].copy()

    total_checks = int(latest["dq_check_name"].nunique())
    failed_checks = int((latest["status"] != "PASS").sum())
    failed_rows_sum = int(latest["failed_rows"].sum())

    by_type = (
        latest.groupby("check_type", as_index=False)
        .agg(
            failed_checks=("status", lambda s: int((s != "PASS").sum())),
            failed_rows=("failed_rows", "sum"),
        )
        .sort_values(["failed_checks", "failed_rows"], ascending=False)
    )

    fig_type = px.bar(by_type, x="check_type", y="failed_checks", title="Failed checks by type")

    top = latest.sort_values("failed_rows", ascending=False).head(25)
    fig_top = px.bar(top, x="failed_rows", y="dq_check_name", orientation="h", title="Top failures")
    fig_top.update_layout(height=780)

    # small 3D chart (top 150) so it never bogs down
    top3d = latest.sort_values("failed_rows", ascending=False).head(150).reset_index(drop=True)
    top3d["check_i"] = top3d.index

    fig_3d = go.Figure(
        data=go.Scatter3d(
            x=top3d["check_i"],
            y=top3d["check_type"],
            z=top3d["failed_rows"],
            mode="markers",
            marker=dict(size=4),
            text=top3d["dq_check_name"],
            hovertemplate="Check=%{text}<br>Type=%{y}<br>Failed rows=%{z}<extra></extra>",
        )
    )
    fig_3d.update_layout(title="3D Failure Severity (top 150)", height=650)

    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Data Quality Dashboard</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        .kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }}
        .kpi {{ border: 1px solid #2b2b2b33; border-radius: 12px; padding: 14px; }}
        .kpi h2 {{ margin: 0; font-size: 28px; }}
        .kpi p {{ margin: 6px 0 0 0; color: #555; }}
        .note {{ color:#666; margin: 6px 0 18px; }}
      </style>
    </head>
    <body>
      <h1>Data Quality Dashboard</h1>
      <div class="note">Generated from <b>{DB_PATH.name}</b> — {latest_ts}</div>

      <div class="kpis">
        <div class="kpi"><h2>{total_checks}</h2><p>Checks executed</p></div>
        <div class="kpi"><h2>{failed_checks}</h2><p>Checks failed</p></div>
        <div class="kpi"><h2>{failed_rows_sum}</h2><p>Total failed rows</p></div>
      </div>

      {fig_type.to_html(full_html=False, include_plotlyjs="cdn")}
      {fig_top.to_html(full_html=False, include_plotlyjs=False)}
      {fig_3d.to_html(full_html=False, include_plotlyjs=False)}
    </body>
    </html>
    """
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote HTML: {OUT_HTML}")


def build_scorecard_png(df: pd.DataFrame) -> None:
    # If kaleido hangs on your machine, we’ll skip PNG and still succeed.
    df = df.copy()
    df["run_ts"] = pd.to_datetime(df["run_ts"])
    df["failed_rows"] = pd.to_numeric(df["failed_rows"], errors="coerce").fillna(0).astype(int)

    latest_ts = df["run_ts"].max()
    latest = df[df["run_ts"] == latest_ts].copy()
    top = latest.sort_values("failed_rows", ascending=False).head(15)

    fig = px.bar(
        top,
        x="failed_rows",
        y="dq_check_name",
        orientation="h",
        title=f"DQ Scorecard (Top failed rows) — {latest_ts:%Y-%m-%d %H:%M}",
    )
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=70, b=20))

    try:
        fig.write_image(str(OUT_PNG), scale=2)
        print(f"Wrote PNG: {OUT_PNG}")
    except Exception as e:
        print("PNG export skipped (kaleido issue). Dashboard HTML is still created.")
        print("Error:", e)


def main() -> None:
    ensure_docs()
    print("Using DB:", DB_PATH)
    con = duckdb.connect(str(DB_PATH))

    df = compute_checks_fast(con)

    build_dashboard(df)
    build_scorecard_png(df)

    print("Done.")


if __name__ == "__main__":
    main()
