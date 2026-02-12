# python-elt-quality-pipeline

A small but complete Python ELT + data-quality pipeline that builds a DuckDB star schema, runs automated DQ checks, and publishes an interactive dashboard via GitHub Pages.

## Live dashboard (GitHub Pages)

[![Open DQ Dashboard](https://img.shields.io/badge/Open%20DQ%20Dashboard-Click%20here-blue)](https://simasaadi.github.io/python-elt-quality-pipeline/)

## What you get

- DuckDB star schema (dimensions + facts; analytics-ready)
- Repeatable ELT run script(s)
- Data-quality checks (nulls, duplicates, foreign-key integrity) with stored history
- DQ dashboard (interactive HTML) + scorecard snapshot for README
- CI workflow (nightly / on-push) to surface regressions

## Repo structure

```text
.github/workflows/     CI workflows (nightly + on push)
data/
  raw/                 sample inputs
  processed/           DuckDB database + derived outputs
docs/                  GitHub Pages site (index.html)
src/                   pipeline code (ELT + marts)
tools/                 utilities (build dashboard, generate viz)
run_all.ps1            Windows end-to-end runner
requirements.txt       Python deps
How it works
1) ELT → DuckDB star schema

The pipeline loads raw data, applies transformations, and produces a clean star schema in DuckDB.

2) Data-quality checks

DQ checks run after ELT and write standardized results into DuckDB (so you keep run history):

Null checks on key columns

Duplicate detection

Foreign key integrity (referential completeness)

3) Dashboard + scorecard

A script generates:

docs/index.html (interactive dashboard served by GitHub Pages)

Run locally (Windows)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

.\run_all.ps1
Generate only the dashboard (if ELT + DQ already ran)
python -u tools/make_readme_viz.py

Output files

docs/index.html (interactive dashboard)

data/processed/*.duckdb (database with star schema + dq results)
