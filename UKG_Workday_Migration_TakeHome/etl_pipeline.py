"""
etl_pipeline.py
---------------
Orchestrator for the full UKG -> Workday ETL pipeline.
Extended from the starter file per the README instructions.

Stages:
  1. Extract + schema check          (inline)
  2. Deterministic cleaning          (cleaning_real.run_cleaning)
  3. SQLite staging                  (inline)
  4. LLM fuzzy mapping               (llm_mapper.apply_llm_mappings)
  5. Workday EIB export              (workday_exporter.run_export)
  6. Streamlit review surface        (app.py -- run separately)
"""

import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

from cleaning_real import run_cleaning
from llm_mapper import apply_llm_mappings
from workday_exporter import run_export

print("=" * 52)
print("Acme Health ETL Pipeline")
print("=" * 52)

CSV_PATH = "ukg_employees_raw.csv"
DB_PATH  = "sqlite:///employees.db"

REQUIRED_COLUMNS = [
    "Employee_ID", "First_Name", "Last_Name", "Birth_Date",
    "Hire_Date", "Gender", "Email", "Job_Title", "Department_Name",
    "Location_Name", "Pay_Type", "Annual_Salary",
]

# ---------------------------------------------------------------------------
# Stage 1 -- Extract + schema check
# ---------------------------------------------------------------------------

print("\n[Stage 1] Loading UKG export...")
raw_check = pd.read_csv(CSV_PATH, nrows=0)
missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_check.columns]
if missing_cols:
    raise ValueError(
        f"Schema check failed -- missing columns: {missing_cols}\n"
        "UKG export may have drifted. Fix source file before proceeding."
    )
print("  Schema check passed.")

# ---------------------------------------------------------------------------
# Stage 2 -- Deterministic cleaning
# ---------------------------------------------------------------------------

print("\n[Stage 2] Deterministic cleaning (cleaning_real.py)...")
clean_df, exceptions_df, dupes_df, stats = run_cleaning(CSV_PATH)
print(f"  Raw: {stats['raw_count']}  |  "
      f"Dupes removed: {stats['dupes_removed']}  |  "
      f"Clean: {stats['clean_count']}  |  "
      f"Exceptions: {stats['exception_count']}")

# ---------------------------------------------------------------------------
# Stage 3 -- SQLite staging
# ---------------------------------------------------------------------------

print("\n[Stage 3] Writing to SQLite (employees.db)...")
engine = create_engine(DB_PATH)
clean_df.to_sql("workday_employees", engine, if_exists="replace", index=False)
exceptions_df.to_sql("exceptions",   engine, if_exists="replace", index=False)
if len(dupes_df):
    dupes_df.to_sql("dupes", engine, if_exists="replace", index=False)
print(f"  workday_employees: {len(clean_df)} rows")
print(f"  exceptions:        {len(exceptions_df)} rows")
print(f"  dupes:             {len(dupes_df)} rows")

# ---------------------------------------------------------------------------
# Stage 4 -- LLM mapping
# ---------------------------------------------------------------------------

print("\n[Stage 4] LLM mapping (llm_mapper.py)...")
print("  Hardcoded lookups run first; LLM called only for unrecognised values.")
clean_df = apply_llm_mappings(clean_df)
needs_review = int(clean_df["needs_human_review"].sum())
print(f"  Auto-mapped:          {len(clean_df) - needs_review}")
print(f"  Flagged for review:   {needs_review}")
clean_df.to_sql("workday_employees", engine, if_exists="replace", index=False)
print("  workday_employees updated with LLM columns.")

# ---------------------------------------------------------------------------
# Stage 5 -- Workday EIB export
# ---------------------------------------------------------------------------

run_export(DB_PATH)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 52)
print("Pipeline complete.")
print(f"  Timestamp:          {datetime.now().isoformat()}")
print(f"  Clean rows in DB:   {len(clean_df)}")
print(f"  Exception rows:     {len(exceptions_df)}")
print(f"  Need human review:  {needs_review}")
print("  Next: streamlit run app.py  to review and approve.")
print("=" * 52)