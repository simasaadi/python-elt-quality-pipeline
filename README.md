
# python-elt-quality-pipeline

A small but complete **Python ELT + data quality pipeline** that builds a **DuckDB star schema**, runs **automated data-quality checks**, and publishes a **DQ dashboard + scorecard** via **GitHub Pages**.

**What you get**
- Star schema in DuckDB (dimensions + facts, analytics-ready)
- Repeatable ELT run script(s)
- Data-quality checks (nulls, duplicates, FK integrity) with stored history
- DQ dashboard (interactive HTML) + static scorecard image for README
- CI workflow (nightly / on-push) to keep the pipeline honest

---

## Live dashboard (GitHub Pages)

[![Open DQ Dashboard](https://img.shields.io/badge/Open%20DQ%20Dashboard-Click%20here-blue)](https://simasaadi.github.io/python-elt-quality-pipeline/dq_dashboard.html)


**DQ Scorecard (latest snapshot)**  
![DQ Scorecard](docs/dq_scorecard.png)

---

## Repo structure

```text
.github/workflows/        CI workflows (nightly + on push)
data/
  raw/                    raw inputs (sample data)
  processed/              DuckDB database + derived outputs
docs/                     GitHub Pages site (index.html + scorecard)
src/                      pipeline code (ELT + marts)
tools/                    utility scripts (build dashboard, generate viz)
run_all.ps1               Windows runner (end-to-end)
requirements.txt

How it works
1) ELT → DuckDB star schema

The pipeline loads raw data, applies transformations, and produces a clean star schema in DuckDB (dimensions + facts) suitable for analytics and reporting.

2) Data-quality checks

DQ checks run after the ELT step and write standardized results into a DuckDB table (so you keep history across runs). Current checks include:

Null checks on key columns

Duplicate detection

Foreign key integrity (referential completeness)

3) Dashboard + scorecard

A script generates:

docs/index.html (interactive dashboard)

docs/dq_scorecard.png (static snapshot for README)

GitHub Pages serves everything inside docs/.
Run locally (Windows)
Prereqs

Python 3.11+ (3.12 works)

pip install -r requirements.txt

End-to-end run
.\run_all.ps1

Generate only the dashboard (if ELT + DQ already ran)
python -u .\tools\make_readme_viz.py


Outputs:

docs/index.html

docs/dq_scorecard.png

CI / Automation

This repo includes GitHub Actions workflows to run the pipeline automatically (nightly +/or on push depending on workflow settings).
The intent is simple: DQ regressions should be visible immediately (via stored results + dashboard updates).






