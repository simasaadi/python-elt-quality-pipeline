from __future__ import annotations

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "processed" / "retail_star_schema.duckdb"

OUT_DIR = REPO_ROOT / "docs"
OUT_INDEX = OUT_DIR / "index.html"

# ---------------------------
# DuckDB helpers
# ---------------------------

def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))

def list_tables(con: duckdb.DuckDBPyConnection) -> List[Tuple[str, str]]:
    rows = con.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type='BASE TABLE'
        ORDER BY 1,2
        """
    ).fetchall()
    return [(r[0], r[1]) for r in rows]

def table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    return con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema=? AND table_name=?
        LIMIT 1
        """,
        [schema, table],
    ).fetchone() is not None

def describe(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> List[Tuple[str, str]]:
    rows = con.execute(f"DESCRIBE {schema}.{table}").fetchall()
    return [(r[0], r[1]) for r in rows]

def cols(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> List[str]:
    return [c for (c, _t) in describe(con, schema, table)]

def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def _is_numeric_type(t: str) -> bool:
    tt = t.lower()
    return any(x in tt for x in ["int", "bigint", "smallint", "tinyint", "double", "real", "float", "decimal", "numeric"])

def _pick_first(candidates: List[str], available: List[str]) -> Optional[str]:
    aset = {a.lower(): a for a in available}
    for c in candidates:
        if c.lower() in aset:
            return aset[c.lower()]
    return None

def _is_keyish(col: str) -> bool:
    c = col.lower()
    return c.endswith("_sk") or c.endswith("_id")

def _pick_date_key(fact_cols: List[str]) -> Optional[str]:
    lc = [c.lower() for c in fact_cols]
    for preferred in ["date_sk", "order_date_sk", "sale_date_sk", "txn_date_sk", "transaction_date_sk"]:
        if preferred in lc:
            return fact_cols[lc.index(preferred)]
    for i, c in enumerate(lc):
        if c.endswith("_date_sk"):
            return fact_cols[i]
    for i, c in enumerate(lc):
        if "date" in c and c.endswith("_sk"):
            return fact_cols[i]
    return None

def ensure_history_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS dq_runs (
          run_id VARCHAR PRIMARY KEY,
          run_ts TIMESTAMP,
          db_name VARCHAR,
          total_checks INTEGER,
          failed_checks INTEGER,
          total_failed_rows BIGINT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS dq_checks (
          run_id VARCHAR,
          run_ts TIMESTAMP,
          status VARCHAR,
          check_type VARCHAR,
          table_name VARCHAR,
          column_name VARCHAR,
          dq_check_name VARCHAR,
          failed_rows BIGINT,
          severity INTEGER
        );
        """
    )

# ---------------------------
# DQ checks
# ---------------------------

def compute_checks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    run_ts = datetime.now()
    tables = list_tables(con)
    mart_tables = [(s, t) for (s, t) in tables if s == "mart"]
    if not mart_tables:
        raise SystemExit(f"No mart schema tables found. Available tables: {tables}")

    checks: List[Dict] = []

    # NULL checks
    print("[1/3] Running NULL checks...")
    for schema, table in mart_tables:
        d = describe(con, schema, table)
        key_cols = [c for (c, _t) in d if _is_keyish(c)]
        for col in key_cols[:25]:
            q = f"SELECT COUNT(*) FROM {schema}.{table} WHERE {_qident(col)} IS NULL"
            nulls = int(con.execute(q).fetchone()[0])
            checks.append({
                "run_ts": run_ts,
                "check_type": "nulls",
                "table": f"{schema}.{table}",
                "column": col,
                "dq_check_name": f"{schema}.{table}__{col}__nulls",
                "status": "PASS" if nulls == 0 else "FAIL",
                "failed_rows": nulls,
                "severity": 3 if nulls > 0 else 0,
            })

    # DUPLICATE checks
    print("[2/3] Running DUPLICATE checks...")
    for schema, table in mart_tables:
        d = describe(con, schema, table)
        key_cols = [c for (c, _t) in d if _is_keyish(c)]
        preferred = _pick_first([c for c in key_cols if c.lower().endswith("_sk")], key_cols) or (key_cols[0] if key_cols else None)
        if not preferred:
            continue

        q = f"""
        SELECT COALESCE(SUM(cnt - 1), 0)
        FROM (
          SELECT {_qident(preferred)} AS k, COUNT(*) AS cnt
          FROM {schema}.{table}
          WHERE {_qident(preferred)} IS NOT NULL
          GROUP BY 1
          HAVING COUNT(*) > 1
        )
        """
        dups = int(con.execute(q).fetchone()[0])
        checks.append({
            "run_ts": run_ts,
            "check_type": "duplicates",
            "table": f"{schema}.{table}",
            "column": preferred,
            "dq_check_name": f"{schema}.{table}__{preferred}__duplicates",
            "status": "PASS" if dups == 0 else "FAIL",
            "failed_rows": dups,
            "severity": 4 if dups > 0 else 0,
        })

    # FK checks (fact -> dim matching on *_sk column names)
    print("[3/3] Running FK checks...")
    dim_tables = [(s, t) for (s, t) in mart_tables if t.startswith("dim_")]
    dim_col_index: Dict[str, List[Tuple[str, str]]] = {}
    for s, t in dim_tables:
        for c, _t in describe(con, s, t):
            dim_col_index.setdefault(c.lower(), []).append((s, t))

    fact_tables = [(s, t) for (s, t) in mart_tables if t.startswith("fct_") or t.startswith("fact_")]
    for schema, table in fact_tables:
        fcols = [c for (c, _t) in describe(con, schema, table)]
        fk_cols = [c for c in fcols if c.lower().endswith("_sk")]
        for fk in fk_cols[:25]:
            targets = dim_col_index.get(fk.lower(), [])
            if not targets:
                continue
            dim_schema, dim_table = sorted(targets)[0]

            q = f"""
            SELECT COUNT(*)
            FROM {schema}.{table} f
            LEFT JOIN {dim_schema}.{dim_table} d
              ON f.{_qident(fk)} = d.{_qident(fk)}
            WHERE f.{_qident(fk)} IS NOT NULL
              AND d.{_qident(fk)} IS NULL
            """
            missing = int(con.execute(q).fetchone()[0])
            checks.append({
                "run_ts": run_ts,
                "check_type": "fk_missing",
                "table": f"{schema}.{table}",
                "column": fk,
                "dq_check_name": f"{schema}.{table}__{fk}__missing_in_{dim_schema}.{dim_table}",
                "status": "PASS" if missing == 0 else "FAIL",
                "failed_rows": missing,
                "severity": 5 if missing > 0 else 0,
            })

    df = pd.DataFrame(checks)
    if df.empty:
        raise SystemExit("No DQ checks produced. Something is wrong with table discovery.")
    return df

def persist_run(con: duckdb.DuckDBPyConnection, dq: pd.DataFrame, db_name: str) -> Tuple[str, datetime]:
    ensure_history_tables(con)

    run_ts = dq["run_ts"].iloc[0].to_pydatetime() if "run_ts" in dq.columns else datetime.now()
    run_id = run_ts.strftime("%Y%m%d%H%M%S%f")

    total_checks = int(len(dq))
    failed_checks = int((dq["status"] == "FAIL").sum())
    total_failed_rows = int(dq.loc[dq["status"] == "FAIL", "failed_rows"].sum())

    con.execute(
        """
        INSERT INTO dq_runs (run_id, run_ts, db_name, total_checks, failed_checks, total_failed_rows)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [run_id, run_ts, db_name, total_checks, failed_checks, total_failed_rows],
    )

    # detail rows
    rows = dq.copy()
    rows["run_id"] = run_id
    rows = rows.rename(columns={"table": "table_name", "column": "column_name"})
    con.register("dq_tmp", rows[[
        "run_id","run_ts","status","check_type","table_name","column_name","dq_check_name","failed_rows","severity"
    ]])
    con.execute("INSERT INTO dq_checks SELECT * FROM dq_tmp;")
    con.unregister("dq_tmp")

    return run_id, run_ts

def fetch_run_history(con: duckdb.DuckDBPyConnection, limit: int = 20) -> pd.DataFrame:
    if not table_exists(con, "main", "dq_runs"):
        return pd.DataFrame()
    q = f"""
    SELECT run_ts, failed_checks, total_failed_rows, total_checks
    FROM dq_runs
    ORDER BY run_ts DESC
    LIMIT {int(limit)}
    """
    df = con.execute(q).df()
    if df.empty:
        return df
    return df.sort_values("run_ts")

# ---------------------------
# Business metrics (safe)
# ---------------------------

def _pick_measure(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> Optional[str]:
    d = describe(con, schema, table)
    candidates = ["revenue","total_amount","sales_amount","amount","net_sales","gross_sales"]
    colnames = [c for (c, _t) in d]
    picked = _pick_first(candidates, colnames)
    if picked:
        return picked
    for c, t in d:
        if _is_numeric_type(t) and not _is_keyish(c):
            return c
    return None

def business_frames(con: duckdb.DuckDBPyConnection) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    if not table_exists(con, "mart", "fct_sales"):
        return out

    fact_cols = cols(con, "mart", "fct_sales")
    measure = _pick_measure(con, "mart", "fct_sales") or "1"
    date_key = _pick_date_key(fact_cols)

    has_dim_dates = table_exists(con, "mart", "dim_dates") and ("date_sk" in [c.lower() for c in cols(con, "mart", "dim_dates")])

    # Trend + heatmap: real join if possible, else synthetic dates (clearly labeled)
    if date_key and has_dim_dates:
        q = f"""
        SELECT
          CAST(strptime(CAST(d.date_sk AS VARCHAR), '%Y%m%d') AS DATE) AS date,
          SUM(CAST(f.{_qident(measure)} AS DOUBLE)) AS metric
        FROM mart.fct_sales f
        JOIN mart.dim_dates d
          ON f.{_qident(date_key)} = d.date_sk
        GROUP BY 1
        ORDER BY 1
        """
        try:
            out["trend"] = con.execute(q).df()
        except Exception:
            pass

        qh = f"""
        SELECT
          STRFTIME(CAST(strptime(CAST(d.date_sk AS VARCHAR), '%Y%m%d') AS DATE), '%b') AS month,
          STRFTIME(CAST(strptime(CAST(d.date_sk AS VARCHAR), '%Y%m%d') AS DATE), '%a') AS dow,
          SUM(CAST(f.{_qident(measure)} AS DOUBLE)) AS metric
        FROM mart.fct_sales f
        JOIN mart.dim_dates d
          ON f.{_qident(date_key)} = d.date_sk
        GROUP BY 1,2
        """
        try:
            out["heatmap"] = con.execute(qh).df()
        except Exception:
            pass
    else:
        # synthetic daily series from row_number() to avoid blank business panels
        try:
            df = con.execute(
                f"""
                SELECT
                  ROW_NUMBER() OVER () AS rn,
                  CAST(CAST(f.{_qident(measure)} AS DOUBLE) AS DOUBLE) AS metric
                FROM mart.fct_sales f
                LIMIT 365
                """
            ).df()
            if not df.empty:
                start = date(2024, 1, 1)
                df["date"] = [start + timedelta(days=int(i)-1) for i in df["rn"]]
                out["trend_synth"] = df[["date","metric"]]
                # make a heatmap from synthetic dates too
                df["month"] = pd.to_datetime(df["date"]).dt.strftime("%b")
                df["dow"] = pd.to_datetime(df["date"]).dt.strftime("%a")
                out["heatmap_synth"] = df.groupby(["month","dow"], as_index=False)["metric"].sum()
        except Exception:
            pass

    # Top stores
    if table_exists(con, "mart", "dim_stores"):
        store_fk = _pick_first(["store_sk","store_id"], fact_cols)
        if store_fk and store_fk.lower().endswith("_sk") and "store_name" in [c.lower() for c in cols(con, "mart", "dim_stores")]:
            q = f"""
            SELECT
              s.store_name AS store,
              SUM(CAST(f.{_qident(measure)} AS DOUBLE)) AS metric
            FROM mart.fct_sales f
            JOIN mart.dim_stores s
              ON f.{_qident(store_fk)} = s.store_sk
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 15
            """
            try:
                out["top_stores"] = con.execute(q).df()
            except Exception:
                pass

    # Revenue mix by category
    if table_exists(con, "mart", "dim_products"):
        prod_fk = _pick_first(["product_sk","product_id"], fact_cols)
        pcols = cols(con, "mart", "dim_products")
        cat_col = _pick_first(["category","product_category","category_name","department","segment"], pcols)
        if prod_fk and prod_fk.lower().endswith("_sk") and cat_col:
            q = f"""
            SELECT
              p.{_qident(cat_col)} AS category,
              SUM(CAST(f.{_qident(measure)} AS DOUBLE)) AS metric
            FROM mart.fct_sales f
            JOIN mart.dim_products p
              ON f.{_qident(prod_fk)} = p.product_sk
            GROUP BY 1
            ORDER BY 2 DESC
            """
            try:
                out["rev_mix"] = con.execute(q).df()
            except Exception:
                pass

    # Campaign perf (scatter bubble)
    if table_exists(con, "mart", "dim_campaigns") and "campaign_sk" in [c.lower() for c in fact_cols]:
        # safest: aggregate directly
        cname = _pick_first(["campaign_name","name"], cols(con, "mart", "dim_campaigns")) or "campaign_sk"
        q = f"""
        SELECT
          c.{_qident(cname)} AS campaign,
          COUNT(*)::DOUBLE AS transactions,
          AVG(CAST(f.{_qident(measure)} AS DOUBLE)) AS avg_order_value,
          SUM(CAST(f.{_qident(measure)} AS DOUBLE)) AS revenue
        FROM mart.fct_sales f
        JOIN mart.dim_campaigns c
          ON f.campaign_sk = c.campaign_sk
        GROUP BY 1
        ORDER BY 4 DESC
        LIMIT 40
        """
        try:
            out["campaign_perf"] = con.execute(q).df()
        except Exception:
            pass

    return out

# ---------------------------
# Figure helpers
# ---------------------------

def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=15, color="#374151")
    )
    fig.update_layout(title=title, height=420, margin=dict(l=20, r=20, t=70, b=20))
    return fig

def _kpi(title: str, value: str, subtitle: str = "") -> str:
    sub = f'<div style="font-size:12px;color:#6b7280;margin-top:6px;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="flex:1; background:#ffffff; border:1px solid #e5e7eb; border-radius:18px; padding:16px 18px; min-width:220px;">
      <div style="font-size:12px; color:#6b7280; margin-bottom:6px;">{title}</div>
      <div style="font-size:34px; font-weight:800; color:#111827; line-height:1;">{value}</div>
      {sub}
    </div>
    """

# ---------------------------
# Dashboard
# ---------------------------

def build_dashboard(
    dq: pd.DataFrame,
    history: pd.DataFrame,
    biz: Dict[str, pd.DataFrame],
    db_name: str,
    run_ts: datetime,
) -> str:
    total_checks = int(len(dq))
    failed_checks = int((dq["status"] == "FAIL").sum())
    total_failed_rows = int(dq.loc[dq["status"] == "FAIL", "failed_rows"].sum())
    pass_rate = (total_checks - failed_checks) / total_checks if total_checks else 1.0

    # DQ status donut
    status_counts = dq.groupby("status", as_index=False).size()
    fig_status = px.pie(status_counts, names="status", values="size", hole=0.55, title="DQ status (latest run)")
    fig_status.update_layout(height=460, margin=dict(l=20, r=20, t=70, b=20))

    # by type
    by_type = dq.groupby(["check_type", "status"], as_index=False).size()
    fig_by_type = px.bar(by_type, x="check_type", y="size", color="status", barmode="stack", title="DQ checks by type (PASS/FAIL)")
    fig_by_type.update_layout(height=460, margin=dict(l=20, r=20, t=70, b=20))

    # coverage matrix
    cov = dq.groupby(["table", "check_type"], as_index=False).agg(n=("dq_check_name", "count"))
    top_tables = cov.groupby("table", as_index=False)["n"].sum().sort_values("n", ascending=False).head(12)["table"].tolist()
    cov = cov[cov["table"].isin(top_tables)]
    if cov.empty:
        fig_cov = _empty_figure("DQ coverage matrix", "No coverage data available.")
    else:
        piv = cov.pivot_table(index="table", columns="check_type", values="n", fill_value=0)
        fig_cov = px.imshow(piv, title="DQ coverage matrix (top tables)", aspect="auto")
        fig_cov.update_layout(height=520, margin=dict(l=20, r=20, t=70, b=20))

    # top failures (never blank)
    failed = dq[dq["status"] == "FAIL"].copy()
    if failed.empty:
        fig_fail = _empty_figure("Top DQ failures", "No failed checks ✅")
    else:
        top = failed.sort_values("failed_rows", ascending=False).head(25)
        fig_fail = px.bar(
            top,
            x="failed_rows",
            y="dq_check_name",
            orientation="h",
            title="Top DQ failures (by failed rows)",
            hover_data=["check_type", "table", "column"]
        )
        fig_fail.update_layout(height=720, margin=dict(l=20, r=20, t=70, b=20))

    # failures over time (needs >=2 runs)
    if history is None or history.empty or len(history) < 2:
        fig_hist = _empty_figure("DQ failures over time (last runs)", "Not enough history yet. Run the script a few times (history is stored in DuckDB).")
    else:
        h = history.copy()
        fig_hist = px.line(h, x="run_ts", y="failed_checks", markers=True, title="DQ failures over time (last runs)")
        fig_hist.update_layout(height=420, margin=dict(l=20, r=20, t=70, b=20))

    # detail table
    detail = dq.sort_values(["status", "failed_rows"], ascending=[True, False]).head(60)
    fig_detail = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["status", "check_type", "table", "column", "failed_rows"],
                    fill_color="#f3f4f6",
                    align="left",
                    font=dict(size=12),
                    height=28,
                ),
                cells=dict(
                    values=[detail["status"], detail["check_type"], detail["table"], detail["column"], detail["failed_rows"]],
                    align="left",
                    font=dict(size=11),
                    height=26,
                ),
            )
        ]
    )
    fig_detail.update_layout(title="DQ check detail (latest)", height=560, margin=dict(l=20, r=20, t=70, b=20))

    # Business figs
    if "trend" in biz and not biz["trend"].empty:
        fig_trend = px.line(biz["trend"], x="date", y="metric", title="Sales trend")
        fig_trend.update_layout(height=460, margin=dict(l=20, r=20, t=70, b=20))
    elif "trend_synth" in biz and not biz["trend_synth"].empty:
        fig_trend = px.line(biz["trend_synth"], x="date", y="metric", title="Sales trend (synthetic dates to enable visuals)")
        fig_trend.update_layout(height=460, margin=dict(l=20, r=20, t=70, b=20))
    else:
        fig_trend = _empty_figure("Sales trend", "Sales trend not available (missing mart.fct_sales or numeric measure).")

    hm_df = None
    hm_title = "Seasonality heatmap (month × day-of-week)"
    if "heatmap" in biz and not biz["heatmap"].empty:
        hm_df = biz["heatmap"]
    elif "heatmap_synth" in biz and not biz["heatmap_synth"].empty:
        hm_df = biz["heatmap_synth"]
        hm_title = "Seasonality heatmap (synthetic dates)"

    if hm_df is not None and not hm_df.empty:
        dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        hm = hm_df.copy()
        hm["dow"] = pd.Categorical(hm["dow"], categories=dow_order, ordered=True)
        hm["month"] = pd.Categorical(hm["month"], categories=month_order, ordered=True)
        piv = hm.pivot_table(index="month", columns="dow", values="metric", aggfunc="sum", fill_value=0)
        fig_heat = px.imshow(piv, title=hm_title, aspect="auto")
        fig_heat.update_layout(height=460, margin=dict(l=20, r=20, t=70, b=20))
    else:
        fig_heat = _empty_figure("Seasonality heatmap", "Not available (need dim_dates join or synthetic fallback).")

    if "top_stores" in biz and not biz["top_stores"].empty:
        ts = biz["top_stores"].copy().sort_values("metric", ascending=True)
        fig_stores = px.bar(ts, x="metric", y="store", orientation="h", title="Top stores (leaderboard)")
        fig_stores.update_layout(height=520, margin=dict(l=20, r=20, t=70, b=20))
    else:
        fig_stores = _empty_figure("Top stores (leaderboard)", "Not available (need dim_stores + store_sk in fct_sales).")

    if "rev_mix" in biz and not biz["rev_mix"].empty:
        rm = biz["rev_mix"].copy()
        rm = rm[rm["metric"] > 0].sort_values("metric", ascending=False).head(12)
        fig_mix = px.treemap(rm, path=["category"], values="metric", title="Revenue mix by category (treemap)")
        fig_mix.update_layout(height=520, margin=dict(l=20, r=20, t=70, b=20))
    else:
        fig_mix = _empty_figure("Revenue mix by category", "Not available (need product category in dim_products).")

    if "campaign_perf" in biz and not biz["campaign_perf"].empty:
        cp = biz["campaign_perf"].copy()
        fig_camp = px.scatter(
            cp,
            x="transactions",
            y="avg_order_value",
            size="revenue",
            hover_name="campaign",
            title="Campaign performance (transactions × AOV, bubble = revenue)"
        )
        fig_camp.update_layout(height=560, margin=dict(l=20, r=20, t=70, b=20))
    else:
        fig_camp = _empty_figure("Campaign performance", "Not available (need dim_campaigns + campaign_sk in fct_sales).")

    kpis = f"""
    <div style="display:flex; gap:14px; flex-wrap:wrap; margin: 16px 0 18px 0;">
      {_kpi("DQ checks executed", str(total_checks), "nulls / duplicates / FK checks")}
      {_kpi("DQ checks failed", str(failed_checks), "goal: 0")}
      {_kpi("Total failed rows", str(total_failed_rows), "across failed checks")}
      {_kpi("DQ pass rate", f"{pass_rate*100:.1f}%", "latest run")}
    </div>
    """

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Python ELT + Quality Pipeline Dashboard</title>
      <style>
        body {{
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          background:#f6f7fb; color:#111827; margin:0;
        }}
        .wrap {{ max-width: 1240px; margin: 0 auto; padding: 18px 18px 40px; }}
        h1 {{ font-size: 22px; margin: 8px 0 6px; font-weight: 800; }}
        .meta {{
          display:flex; gap:10px; align-items:center; flex-wrap:wrap;
          color:#6b7280; font-size: 13px; margin-bottom: 10px;
        }}
        .pill {{
          background:#fff; border:1px solid #e5e7eb; border-radius:999px; padding:6px 10px;
        }}
        .grid2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
        @media (max-width: 980px) {{
          .grid2 {{ grid-template-columns: 1fr; }}
        }}
        .card {{
          background:#fff; border:1px solid #e5e7eb; border-radius:18px; padding:14px; margin: 14px 0;
          box-shadow: 0 2px 10px rgba(17,24,39,0.04);
        }}
        .tabs {{
          display:flex; gap:10px; margin: 10px 0 0;
        }}
        .tab {{
          padding: 8px 12px; border-radius:999px; border:1px solid #e5e7eb;
          background:#fff; font-size: 13px; font-weight: 700; color:#111827;
        }}
        .tab.active {{
          background:#111827; color:#fff; border-color:#111827;
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <h1>Python ELT + Quality Pipeline Dashboard</h1>
        <div class="meta">
          <span class="pill">Generated from <b>{db_name}</b></span>
          <span class="pill">Run: <b>{run_ts}</b></span>
        </div>

        {kpis}

        <div class="tabs">
          <div class="tab active">Business</div>
          <div class="tab">Data Quality</div>
        </div>

        <div class="grid2">
          <div class="card">{fig_trend.to_html(full_html=False, include_plotlyjs="cdn")}</div>
          <div class="card">{fig_heat.to_html(full_html=False, include_plotlyjs=False)}</div>
        </div>

        <div class="grid2">
          <div class="card">{fig_stores.to_html(full_html=False, include_plotlyjs=False)}</div>
          <div class="card">{fig_mix.to_html(full_html=False, include_plotlyjs=False)}</div>
        </div>

        <div class="card">{fig_camp.to_html(full_html=False, include_plotlyjs=False)}</div>

        <div class="tabs" style="margin-top:22px;">
          <div class="tab active">Data Quality</div>
          <div class="tab">Business</div>
        </div>

        <div class="grid2">
          <div class="card">{fig_status.to_html(full_html=False, include_plotlyjs=False)}</div>
          <div class="card">{fig_by_type.to_html(full_html=False, include_plotlyjs=False)}</div>
        </div>

        <div class="card">{fig_hist.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="card">{fig_cov.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="card">{fig_fail.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="card">{fig_detail.to_html(full_html=False, include_plotlyjs=False)}</div>

      </div>
    </body>
    </html>
    """
    return html

def write_outputs(html: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.write_text(html, encoding="utf-8")
    print(f"Wrote HTML: {OUT_INDEX}")

def main() -> None:
    db_path = DEFAULT_DB
    if not db_path.exists():
        raise SystemExit(f"DuckDB not found at: {db_path}")

    print(f"Using DB: {db_path.resolve()}")

    con = connect(db_path)
    dq = compute_checks(con)

    # persist history so "failures over time" becomes real
    run_id, run_ts = persist_run(con, dq, db_path.name)
    history = fetch_run_history(con, limit=30)

    biz = business_frames(con)

    html = build_dashboard(
        dq=dq,
        history=history,
        biz=biz,
        db_name=db_path.name,
        run_ts=run_ts,
    )

    write_outputs(html)
    print(f"Done. run_id={run_id}")

if __name__ == "__main__":
    main()
