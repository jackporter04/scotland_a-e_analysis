"""Build the DuckDB warehouse from the raw PHS CSVs.

Usage:
    python -m src.pipeline

Reads the saved CSVs in data/raw, loads them into DuckDB, runs the SQL files in sql/ in order to
produce the staging and mart tables, then fits the forecast models.
"""
from pathlib import Path

import duckdb

from src import forecast

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SQL = ROOT / "sql"
DB_PATH = ROOT / "data" / "processed" / "ae.duckdb"


# Raw table name -> source file in data/raw (downloaded from PHS open data).
SOURCES = {
    "raw_weekly": "weekly_ae_activity_20260913.csv",
    "raw_boards": "hb14_hb19.csv",
    "raw_hospitals": "hospitals.csv",
    "raw_delayed": "2026-07_delayed-discharge-beddays-health-board.csv",
    "raw_beds": "beds_by_nhs_board_of_treatment_and_specialty.csv",
}


def load_raw(con: duckdb.DuckDBPyConnection) -> None:
    """Load each source CSV unchanged; all cleaning happens in the SQL files."""
    for table, filename in SOURCES.items():
        path = RAW / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        print(f"Loading {filename}")
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{path.as_posix()}')")


def run_sql(con: duckdb.DuckDBPyConnection) -> None:
    for path in sorted(SQL.glob("*.sql")):
        print(f"Running {path.name}")
        con.execute(path.read_text())


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DB_PATH)) as con:
        load_raw(con)
        run_sql(con)
        for table in ("stg_weekly", "mart_weekly_national", "mart_weekly_board", "mart_hospital_league",
                      "mart_monthly_board_drivers", "mart_quarterly_board_occupancy"):
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n:,} rows")

    forecast.main()
    print(f"Built {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
