from __future__ import annotations

from pathlib import Path
import duckdb

DB_PATH = Path("data/processed/retail_star_schema.duckdb")

def main() -> None:
    print("DB:", DB_PATH.resolve())
    print("Size:", DB_PATH.stat().st_size)

    con = duckdb.connect(str(DB_PATH))

    print("\nSHOW TABLES:")
    print(con.execute("SHOW TABLES").fetchall())

    print("\nSCHEMAS:")
    print(con.execute("SELECT schema_name FROM information_schema.schemata ORDER BY 1").fetchall())

    print("\nALL TABLES (schema, name):")
    print(con.execute("SELECT table_schema, table_name FROM information_schema.tables ORDER BY 1,2").fetchall())

    print("\nDQ-LIKE TABLES:")
    print(
        con.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_name ILIKE '%dq%'
               OR table_name ILIKE '%quality%'
               OR table_name ILIKE '%check%'
            ORDER BY 1,2
            """
        ).fetchall()
    )

if __name__ == "__main__":
    main()
