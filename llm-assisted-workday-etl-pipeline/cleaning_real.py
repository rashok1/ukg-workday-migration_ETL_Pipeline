"""
cleaning_real.py
----------------
Stage 2: deterministic cleaning functions only.

Rules:
  - This file contains ONLY function definitions.
  - No top-level execution. Importing this file does nothing.
  - etl_pipeline.py calls run_cleaning() explicitly.

Every field here is owner=RULE per the field-mapping doc.
The LLM never runs here.
"""

import re
import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser
from datetime import datetime


# =========================================================
# INDIVIDUAL CLEANING FUNCTIONS
# Each is independently testable and importable.
# =========================================================

def clean_name(value) -> str | None:
    """Trim, title-case, strip non-ASCII control characters."""
    if pd.isnull(value):
        return None
    return re.sub(r'[^\x20-\x7E]', '', str(value)).strip().title()


def clean_email(value) -> str | None:
    """Lowercase and strip whitespace."""
    if pd.isnull(value):
        return None
    return str(value).strip().lower()


def validate_email(value) -> bool:
    """Returns True if value matches basic email pattern."""
    if not value:
        return False
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(value)))


def parse_date(value):
    """
    Parses mixed UKG date formats: MM-DD-YYYY, MM/DD/YYYY,
    YYYY-MM-DD, YYYY/MM/DD, and literal "invalid".
    Returns a Python date or None -- never NaT -- so SQLite stores NULL.
    """
    if pd.isnull(value):
        return None
    try:
        return dateutil_parser.parse(str(value)).date()
    except Exception:
        return None


def clean_salary(value) -> float:
    """
    Strips "$" and "," then coerces to float.
    Anything non-numeric -> NaN.
    """
    if pd.isnull(value):
        return np.nan
    cleaned = re.sub(r'[$,]', '', str(value).strip())
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def strip_location_suffix(value) -> str:
    """
    Strips trailing 'HQ' or 'Office' from location strings.
    fillna("") before regex so nulls never break str methods on
    larger datasets.
    """
    if pd.isnull(value):
        return "Remote"
    return re.sub(r'\s*(HQ|Office)\b', '', str(value), flags=re.IGNORECASE).strip()


def build_exception_reason(row) -> str:
    """
    Returns a semicolon-separated string of validation failures
    for a single row. Empty string means the row is clean.
    Categories match the field-mapping doc exception table.
    """
    reasons = []
    if not row["salary_valid"]:
        reasons.append("invalid salary")
    if not row["birth_date_valid"]:
        reasons.append("invalid/missing birth date")
    if not row["hire_date_valid"]:
        reasons.append("invalid/missing hire date")
    if pd.notnull(row["Email"]) and not row["email_valid"]:
        reasons.append("invalid email format")
    if row["Department_Name"] == "Unknown":
        reasons.append("missing department -- LLM retry or HUMAN")
    return "; ".join(reasons)


# =========================================================
# MAIN ENTRY POINT
# Called by etl_pipeline.py -- never runs on import.
# =========================================================

def run_cleaning(csv_path: str):
    """
    Loads the UKG CSV, applies all deterministic RULE transformations,
    deduplicates, splits clean vs exception rows, adds audit columns.

    Parameters
    ----------
    csv_path : str
        Path to the raw UKG employee CSV export.

    Returns
    -------
    clean_df      : pd.DataFrame  -- rows ready for Stage 4 LLM mapping
    exceptions_df : pd.DataFrame  -- rows that failed validation
    dupes_df      : pd.DataFrame  -- duplicate rows (logged, not loaded)
    stats         : dict          -- row counts for pipeline summary
    """
    df = pd.read_csv(csv_path)
    raw_count = len(df)

    # --- Names (RULE) ---
    df["First_Name"] = df["First_Name"].apply(clean_name)
    df["Last_Name"]  = df["Last_Name"].apply(clean_name)

    # --- Email (RULE) ---
    df["Email"]       = df["Email"].apply(clean_email)
    df["email_valid"] = df["Email"].apply(validate_email)

    # --- Dates (RULE) ---
    df["Birth_Date"] = df["Birth_Date"].apply(parse_date)
    df["Hire_Date"]  = df["Hire_Date"].apply(parse_date)

    # --- Gender (RULE): M/F -> Male/Female, blank -> Not Specified ---
    gender_map = {"M": "Male", "F": "Female", "m": "Male", "f": "Female",
                  "Male": "Male", "Female": "Female"}
    df["Gender"] = df["Gender"].map(gender_map).fillna("Not Specified")

    # --- Location suffix strip (RULE) ---
    # fillna("") prevents silent null crashes on larger datasets.
    # LLM maps the clean city name -> code in Stage 4, not here.
    df["Location_Name"] = (
        df["Location_Name"]
        .fillna("")
        .apply(strip_location_suffix)
    )

    # --- Department (RULE) ---
    df["Department_Name"] = df["Department_Name"].fillna("Unknown").str.strip()

    # --- Salary (RULE) ---
    df["Annual_Salary"] = df["Annual_Salary"].apply(clean_salary)

    # --- Pay_Type (RULE): blank -> "Salary" ---
    df["Pay_Type"] = df["Pay_Type"].fillna("Salary")

    # --- Deduplication (RULE) ---
    # Keep all copies in dupes_df for audit; remove all but first from main df.
    before_dedup = len(df)
    dupes_df = df[
        df.duplicated(subset=["First_Name", "Last_Name", "Birth_Date"], keep=False)
    ].copy()
    df = df.drop_duplicates(subset=["First_Name", "Last_Name", "Birth_Date"])
    dupes_removed = before_dedup - len(df)

    # --- Validation flags ---
    df["salary_valid"]     = df["Annual_Salary"].notnull()
    df["birth_date_valid"] = df["Birth_Date"].notnull()
    df["hire_date_valid"]  = df["Hire_Date"].notnull()

    # --- Exception split ---
    df["exception_reason"] = df.apply(build_exception_reason, axis=1)
    exceptions_df = df[df["exception_reason"] != ""].copy()
    clean_df      = df[df["exception_reason"] == ""].copy()

    # --- Audit columns (field-mapping doc audit fields table) ---
    timestamp = datetime.now().isoformat()
    for frame in [clean_df, exceptions_df]:
        frame["source_system"]    = "UKG"
        frame["source_record_id"] = frame["Employee_ID"]
        frame["load_timestamp"]   = timestamp
        frame["pipeline_version"] = "v1.1"

    stats = {
        "raw_count":       raw_count,
        "dupes_removed":   dupes_removed,
        "clean_count":     len(clean_df),
        "exception_count": len(exceptions_df),
    }

    return clean_df, exceptions_df, dupes_df, stats