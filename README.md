
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

➡️ **Open DQ Dashboard:** https://simasaadi.github.io/python-elt-quality-pipeline/

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


\## Data Quality






