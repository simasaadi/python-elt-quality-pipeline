import duckdb

con = duckdb.connect("data/processed/retail_star_schema.duckdb")

print("mart.fct_sales columns:")
for row in con.execute("PRAGMA table_info('mart.fct_sales')").fetchall():
    # row = (cid, name, type, notnull, dflt_value, pk)
    print(f"- {row[1]} ({row[2]})")
