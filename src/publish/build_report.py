from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DQ_PATH = REPO_ROOT / "outputs" / "reports" / "dq_results.json"
DOCS_INDEX = REPO_ROOT / "docs" / "index.html"


def main() -> None:
    if not DQ_PATH.exists():
        raise FileNotFoundError(f"DQ results not found. Run checks first: {DQ_PATH}")

    data = json.loads(DQ_PATH.read_text(encoding="utf-8"))
    results = data["results"]
    summary = data["summary"]

    rows_html = []
    for r in results:
        badge = "PASS" if r["status"] == "PASS" else "FAIL"
        rows_html.append(
            f"""
            <tr>
              <td>{r['check_name']}</td>
              <td><strong>{badge}</strong></td>
              <td style="text-align:right;">{r['failed_rows']}</td>
              <td style="text-align:right;">{r['threshold_failed_rows']}</td>
              <td>{r['details']}</td>
            </tr>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Retail Star Schema ELT - Data Quality Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{ color: #555; margin-bottom: 18px; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; min-width: 160px; }}
    .k {{ color: #666; font-size: 12px; margin-bottom: 6px; }}
    .v {{ font-size: 22px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e5e5; padding: 10px; vertical-align: top; }}
    th {{ background: #f7f7f7; text-align: left; }}
  </style>
</head>
<body>
  <h1>Retail Star Schema ELT - Data Quality Report</h1>
  <div class="meta">
    Run UTC: {data.get('run_utc')}<br/>
    DB: {data.get('db_path')}
  </div>

  <div class="cards">
    <div class="card"><div class="k">Total checks</div><div class="v">{summary['total']}</div></div>
    <div class="card"><div class="k">Passed</div><div class="v">{summary['passed']}</div></div>
    <div class="card"><div class="k">Failed</div><div class="v">{summary['failed']}</div></div>
    <div class="card"><div class="k">Generated</div><div class="v">{datetime.now(timezone.utc).strftime('%Y-%m-%d')}</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Check</th>
        <th>Status</th>
        <th>Failed rows</th>
        <th>Threshold</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""

    DOCS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    DOCS_INDEX.write_text(html, encoding="utf-8")
    print("✅ Wrote HTML report")
    print(f"Open: {DOCS_INDEX}")


if __name__ == "__main__":
    main()
