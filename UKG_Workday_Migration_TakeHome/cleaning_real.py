import pandas as pd
import numpy as np
import re

from dateutil import parser
from datetime import datetime
from sqlalchemy import create_engine

# =========================
# LOAD CSV
# =========================

df = pd.read_csv("ukg_employees_raw.csv")

# =========================
# COPY RAW DATA
# =========================

raw_df = df.copy()

# =========================
# CLEANING FUNCTIONS
# =========================

def clean_name(name):

    if pd.isnull(name):
        return None

    return (
        str(name)
        .strip()
        .title()
    )


def clean_email(email):

    if pd.isnull(email):
        return None

    return (
        str(email)
        .strip()
        .lower()
    )


def validate_email(email):

    if pd.isnull(email):
        return False

    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

    return bool(re.match(pattern, email))


def parse_date(date_value):

    if pd.isnull(date_value):
        return None

    try:
        return parser.parse(
            str(date_value)
        ).date()

    except:
        return None


def clean_salary(value):

    if pd.isnull(value):
        return np.nan

    value = str(value).strip().lower()

    value = value.replace("$", "")
    value = value.replace(",", "")

    try:
        return float(value)

    except:
        return np.nan


# =========================
# CLEAN NAMES
# =========================

df["First_Name"] = df["First_Name"].apply(clean_name)

df["Last_Name"] = df["Last_Name"].apply(clean_name)

# =========================
# CLEAN EMAILS
# =========================

df["Email"] = df["Email"].apply(clean_email)

df["email_valid"] = df["Email"].apply(validate_email)

# =========================
# CLEAN DATES
# =========================

df["Birth_Date"] = df["Birth_Date"].apply(parse_date)

df["Hire_Date"] = df["Hire_Date"].apply(parse_date)

# =========================
# CLEAN GENDER
# =========================

gender_map = {
    "M": "Male",
    "F": "Female",
    "Male": "Male",
    "Female": "Female"
}

df["Gender"] = (
    df["Gender"]
    .map(gender_map)
    .fillna("Not Specified")
)

# =========================
# CLEAN LOCATIONS
# =========================

df["Location_Name"] = (
    df["Location_Name"]
    .str.replace(
        r"HQ|Office",
        "",
        regex=True
    )
    .str.strip()
)

# =========================
# CLEAN DEPARTMENT
# =========================

df["Department_Name"] = (
    df["Department_Name"]
    .fillna("Unknown")
    .str.strip()
)

# =========================
# CLEAN SALARY
# =========================

df["Annual_Salary"] = (
    df["Annual_Salary"]
    .apply(clean_salary)
)

# =========================
# REMOVE DUPLICATES
# =========================

before_count = len(df)

df = df.drop_duplicates(
    subset=[
        "First_Name",
        "Last_Name",
        "Birth_Date"
    ]
)

after_count = len(df)

duplicates_removed = before_count - after_count

# =========================
# VALIDATION FLAGS
# =========================

df["salary_valid"] = (
    df["Annual_Salary"]
    .notnull()
)

df["birth_date_valid"] = (
    df["Birth_Date"]
    .notnull()
)

# =========================
# BUILD EXCEPTION REASONS
# =========================

exception_conditions = []

for index, row in df.iterrows():

    reasons = []

    if not row["salary_valid"]:
        reasons.append("Invalid salary")

    if not row["birth_date_valid"]:
        reasons.append("Invalid or missing birth date")

    if pd.notnull(row["Email"]) and not row["email_valid"]:
        reasons.append("Invalid email")

    exception_conditions.append(
        "; ".join(reasons)
    )

df["exception_reason"] = exception_conditions

# =========================
# SPLIT CLEAN VS EXCEPTIONS
# =========================

exceptions_df = df[
    df["exception_reason"] != ""
].copy()

clean_df = df[
    df["exception_reason"] == ""
].copy()

# =========================
# ADD AUDIT COLUMNS
# =========================

timestamp = datetime.now()

clean_df["source_system"] = "UKG"

clean_df["load_timestamp"] = timestamp

clean_df["pipeline_version"] = "v1.0"

exceptions_df["source_system"] = "UKG"

exceptions_df["load_timestamp"] = timestamp

exceptions_df["pipeline_version"] = "v1.0"

clean_df["source_record_id"] = clean_df["Employee_ID"]

exceptions_df["source_record_id"] = exceptions_df["Employee_ID"]

# =========================
# SAVE TO SQLITE
# =========================

from sqlalchemy import create_engine

engine = create_engine(
    "sqlite:///employees.db"
)

clean_df.to_sql(
    "workday_employees",
    engine,
    if_exists="replace",
    index=False
)

exceptions_df.to_sql(
    "exceptions",
    engine,
    if_exists="replace",
    index=False
)

# =========================
# SUMMARY
# =========================

print("\n===== PIPELINE SUMMARY =====")

print(f"Original rows: {len(raw_df)}")

print(f"Duplicates removed: {duplicates_removed}")

print(f"Clean rows: {len(clean_df)}")

print(f"Exception rows: {len(exceptions_df)}")

print("\nDone.")