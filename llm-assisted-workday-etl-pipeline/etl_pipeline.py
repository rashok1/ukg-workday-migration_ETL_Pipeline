"""
etl_pipeline.py
---------------
Orchestration only. No cleaning logic, no LLM logic, no SQL logic here.

  1. SINGLE DB WRITE -- early save_to_db() call removed. The DB is written
     exactly once, AFTER LLM mapping, so workday_employees always contains
     the full final schema (LLM columns + review_status + needs_human_review)
     when app.py reads it. No partial state visible to Streamlit.

  2. SAFE COLUMN GUARDS -- nunique() calls are guarded with `if col in df`
     so the pipeline doesn't crash if a column is missing.

  3. CONSISTENT ISO TIMESTAMPS -- datetime.now().isoformat() used everywhere.
     No mixing of datetime objects and strings.

  4. STABLE TABLE NAMES -- workday_employees is only written once (final).
     exceptions and dupes are written once. No table is overwritten mid-run.

Run:
    python etl_pipeline.py
"""

import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

from cleaning_real import run_cleaning
from llm_mapper import apply_llm_mappings
from workday_exporter import run_export


# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "ukg_employees_raw.csv"
DB_PATH  = "sqlite:///employees.db"

REQUIRED_COLUMNS = [
    "Employee_ID", "First_Name", "Last_Name", "Birth_Date",
    "Hire_Date", "Gender", "Email", "Job_Title", "Department_Name",
    "Location_Name", "Pay_Type", "Annual_Salary",
]


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline():

    print("=" * 52)
    print("Acme Health ETL Pipeline")
    print("=" * 52)

    engine = create_engine(DB_PATH)

    # ---------------------------------------------------
    # Stage 1 -- Extract + schema check
    # ---------------------------------------------------
    print("\n[Stage 1] Schema check...")
    raw_check = pd.read_csv(CSV_PATH, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_check.columns]
    if missing:
        raise ValueError(
            f"Schema check failed -- missing columns: {missing}\n"
            "UKG export may have drifted. Fix source before proceeding."
        )
    print("  Passed.")

    # ---------------------------------------------------
    # Stage 2 -- Deterministic cleaning
    # ---------------------------------------------------
    print("\n[Stage 2] Deterministic cleaning...")
    clean_df, exceptions_df, dupes_df, stats = run_cleaning(CSV_PATH)
    print(f"  Raw: {stats['raw_count']}  |  "
          f"Dupes removed: {stats['dupes_removed']}  |  "
          f"Clean: {stats['clean_count']}  |  "
          f"Exceptions: {stats['exception_count']}")

    # ---------------------------------------------------
    # Stage 3 -- SQLite: exceptions + dupes only
    # workday_employees is NOT written here.
    # It is written once, after Stage 4, with the full schema.
    # This prevents app.py from ever reading a table that is
    # missing LLM columns or needs_human_review.
    # ---------------------------------------------------
    print("\n[Stage 3] Writing exceptions and dupes to SQLite...")
    exceptions_df.to_sql(
        "exceptions", engine, if_exists="replace", index=False
    )
    if len(dupes_df):
        dupes_df.to_sql(
            "dupes", engine, if_exists="replace", index=False
        )
    print(f"  exceptions: {len(exceptions_df)} rows")
    print(f"  dupes:      {len(dupes_df)} rows")
    print("  workday_employees: deferred until after LLM mapping.")

    # ---------------------------------------------------
    # Stage 4 -- LLM mapping (unique values only)
    # needs_human_review is defined HERE in apply_llm_mappings(),
    # never in cleaning_real.py.
    # ---------------------------------------------------
    print("\n[Stage 4] LLM mapping...")

    # Safe column guards -- guard nunique() in case column is missing
    unique_jobs  = clean_df["Job_Title"].nunique()       if "Job_Title"       in clean_df.columns else 0
    unique_depts = clean_df["Department_Name"].nunique() if "Department_Name" in clean_df.columns else 0
    unique_locs  = clean_df["Location_Name"].nunique()   if "Location_Name"   in clean_df.columns else 0

    print(f"  Unique job titles:   {unique_jobs}  (not {len(clean_df)} rows)")
    print(f"  Unique departments:  {unique_depts}")
    print(f"  Unique locations:    {unique_locs}")

    clean_df = apply_llm_mappings(clean_df)

    # needs_human_review is now guaranteed to exist
    needs_review = int(clean_df["needs_human_review"].sum())
    print(f"  Auto-mapped:         {len(clean_df) - needs_review}")
    print(f"  Flagged for review:  {needs_review}")

    # ---------------------------------------------------
    # Stage 3 (final) -- Write workday_employees ONCE,
    # with complete schema including all LLM columns.
    # This is the only write to this table in the pipeline.
    # ---------------------------------------------------
    print("\n[Stage 3-final] Writing workday_employees to SQLite...")
    clean_df.to_sql(
        "workday_employees", engine, if_exists="replace", index=False
    )
    print(f"  workday_employees: {len(clean_df)} rows (full schema, LLM columns included)")

    # ---------------------------------------------------
    # Stage 5 -- Workday EIB export
    # ---------------------------------------------------
    run_export(DB_PATH)

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------
    print("\n" + "=" * 52)
    print("Pipeline complete.")
    print(f"  Timestamp:          {datetime.now().isoformat()}")
    print(f"  Clean rows in DB:   {len(clean_df)}")
    print(f"  Exception rows:     {len(exceptions_df)}")
    print(f"  Need human review:  {needs_review}")
    print("  Next: streamlit run app.py")
    print("=" * 52)


if __name__ == "__main__":
    run_pipeline()