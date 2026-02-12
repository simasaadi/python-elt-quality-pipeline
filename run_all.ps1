$ErrorActionPreference = "Stop"

python -m src.ingest.load_kaggle_raw
python -m src.transform.build_marts
python -m src.validate.run_checks
python -m src.publish.build_report

Write-Host "Done. Open docs\index.html" -ForegroundColor Green
