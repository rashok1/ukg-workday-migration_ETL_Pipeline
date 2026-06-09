"""
llm_mapper.py
-------------
Stage 4: LLM-based fuzzy mapping + persistent in-process cache.

  1. needs_human_review is defined HERE, in apply_llm_mappings().
     It is NEVER referenced in cleaning_real.py (which has no LLM columns).
     etl_pipeline.py reads it from clean_df AFTER this function returns.

  2. Persistent mapping cache via SQLite (mapping_cache table).
     Resolution order per call:
       a. Hardcoded shorthand dict  -> confidence 1.0, no I/O
       b. Catalog key dict          -> confidence 1.0, no I/O
       c. mapping_cache table       -> confidence from prior run, no Ollama
       d. Ollama LLM call           -> result written to cache for next run
     @lru_cache is kept as an in-process safety net on top.

  3. Unique-value mapping in apply_llm_mappings() -- N unique values,
     not N rows. pandas .map() joins results back to full dataframe.

  4. Timestamp standardised to datetime.now().isoformat() everywhere.

"""

import re
import json
import ollama
import sqlite3
from functools import lru_cache
from datetime import datetime


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME           = "llama3:latest"
CONFIDENCE_THRESHOLD = 0.8
CACHE_DB_PATH        = "employees.db"   # same DB as main pipeline


# =========================================================
# CATALOGS  (source: UKG_to_Workday_Field_Mapping.md)
# =========================================================

JOB_CATALOG = {
    "PHYS_FAM":    "Family Physician (MD/DO)",
    "PHYS_INTERN": "Internal Medicine Physician",
    "NP_FAM":      "Nurse Practitioner -- Family",
    "RN":          "Registered Nurse",
    "LPN":         "Licensed Practical Nurse",
    "MA":          "Medical Assistant",
    "PHARM_RX":    "Pharmacist",
    "PHARM_TECH":  "Pharmacy Technician",
    "DENT_DDS":    "Dentist",
    "DENT_HYG":    "Dental Hygienist",
    "BH_LCSW":     "Behavioral Health Counselor (LCSW)",
    "ADMIN_FD":    "Front Desk / Patient Coordinator",
    "ADMIN_PM":    "Practice Manager",
    "ADMIN_CMO":   "Chief Medical Officer",
    "BILL_SPEC":   "Billing Specialist",
    "BILL_CODER":  "Medical Coder",
    "IT_SUP":      "IT Support",
    "HR_SPEC":     "HR Specialist",
    "FIN_ANALYST": "Finance Analyst",
    "OPS_MGR":     "Operations Manager",
}

JOB_SHORTHAND = {
    "RN":                       "RN",
    "REGISTERED NURSE":         "RN",
    "LPN":                      "LPN",
    "LICENSED PRACTICAL NURSE": "LPN",
    "MA":                       "MA",
    "MEDICAL ASSISTANT":        "MA",
    "NP":                       "NP_FAM",
    "NURSE PRACTITIONER":       "NP_FAM",
    "MD":                       "PHYS_FAM",
    "DO":                       "PHYS_FAM",
    "PHARMACIST":               "PHARM_RX",
    "PHARMACY TECHNICIAN":      "PHARM_TECH",
    "PHARM TECH":               "PHARM_TECH",
    "DENTIST":                  "DENT_DDS",
    "DDS":                      "DENT_DDS",
    "DENTAL HYGIENIST":         "DENT_HYG",
    "CMO":                      "ADMIN_CMO",
    "CHIEF MEDICAL OFFICER":    "ADMIN_CMO",
    "BILLING SPECIALIST":       "BILL_SPEC",
    "MEDICAL CODER":            "BILL_CODER",
    "IT SUPPORT":               "IT_SUP",
    "HR SPECIALIST":            "HR_SPEC",
    "FINANCE ANALYST":          "FIN_ANALYST",
    "OPERATIONS MANAGER":       "OPS_MGR",
    "PRACTICE MANAGER":         "ADMIN_PM",
    "FRONT DESK":               "ADMIN_FD",
    "PATIENT COORDINATOR":      "ADMIN_FD",
}

LOCATION_CODES = {
    "AUG":  "Augusta",
    "BAT":  "Batesville",
    "SRCY": "Searcy",
    "HBR":  "Heber Springs",
    "MTN":  "Mountain View",
    "NWP":  "Newport",
    "CBT":  "Cabot",
    "BEE":  "Beebe",
    "CLN":  "Clinton",
    "CRN":  "Corning",
    "REM":  "Remote / Telehealth",
}

LOCATION_SHORTHAND = {v.upper(): k for k, v in LOCATION_CODES.items()}
LOCATION_SHORTHAND["REMOTE"]     = "REM"
LOCATION_SHORTHAND["TELEHEALTH"] = "REM"

DEPT_SHORTHAND = {
    "HR":                     "Human Resources",
    "HUMAN RESOURCES":        "Human Resources",
    "PEOPLE":                 "Human Resources",
    "IT":                     "IT",
    "INFORMATION TECHNOLOGY": "IT",
    "FINANCE":                "Finance",
    "BILLING":                "Billing",
    "ADMIN":                  "Administration",
    "ADMINISTRATION":         "Administration",
    "OPERATIONS":             "Operations",
    "OPS":                    "Operations",
    "PHARMACY":               "Pharmacy",
    "PRIMARY CARE":           "Primary Care",
    "NURSING":                "Primary Care",
    "DENTAL":                 "Dental",
    "BEHAVIORAL HEALTH":      "Behavioral Health",
    "BH":                     "Behavioral Health",
    "UNKNOWN":                "Unknown",
}

ALLOWED_DEPARTMENTS = sorted(set(DEPT_SHORTHAND.values()))


# =========================================================
# PERSISTENT CACHE  (SQLite mapping_cache table)
# Survives between pipeline runs. Ollama is only called
# when a value has never been seen before.
# =========================================================

def _ensure_cache_table():
    """Create mapping_cache table if it doesn't exist."""
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mapping_cache (
                field_type    TEXT NOT NULL,
                raw_value     TEXT NOT NULL,
                workday_value TEXT,
                confidence    REAL,
                reason        TEXT,
                cached_at     TEXT,
                PRIMARY KEY (field_type, raw_value)
            )
        """)


def _cache_get(field_type: str, raw_value: str) -> dict | None:
    """Return cached result or None if not found."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            row = conn.execute(
                "SELECT workday_value, confidence, reason FROM mapping_cache "
                "WHERE field_type = ? AND raw_value = ?",
                (field_type, raw_value)
            ).fetchone()
        if row:
            return {"workday_value": row[0], "confidence": row[1], "reason": row[2]}
        return None
    except Exception:
        return None


def _cache_set(field_type: str, raw_value: str, result: dict):
    """Write a result to the persistent cache."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO mapping_cache "
                "(field_type, raw_value, workday_value, confidence, reason, cached_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (field_type, raw_value,
                 result.get("workday_value"),
                 result.get("confidence", 0.0),
                 result.get("reason", ""),
                 datetime.now().isoformat())
            )
    except Exception:
        pass  # cache write failure is non-fatal


# =========================================================
# CORE LLM CALL
# =========================================================

def _call_ollama(prompt: str) -> dict:
    """
    Single HTTP round-trip to local Ollama.
    Strips markdown fences -- llama3 adds ```json ... ``` even when told
    not to, causing json.loads() to throw and returning confidence 0.0
    silently on every call without this fix.
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"workday_value": None, "confidence": 0.0,
                "reason": "model returned non-JSON output"}
    except Exception as e:
        return {"workday_value": None, "confidence": 0.0,
                "reason": f"LLM error: {e}"}


# =========================================================
# INDIVIDUAL MAPPERS
# Resolution order:
#   1. hardcoded shorthand (no I/O)
#   2. catalog key (no I/O)
#   3. persistent cache read (SQLite)
#   4. Ollama call + cache write
#   @lru_cache is kept as in-process safety net
# =========================================================

@lru_cache(maxsize=256)
def normalize_job_title(job_title: str) -> dict:
    key = job_title.strip().upper()

    if key in JOB_SHORTHAND:
        return {"workday_value": JOB_SHORTHAND[key],
                "confidence": 1.0, "reason": "exact shorthand match"}
    if key in JOB_CATALOG:
        return {"workday_value": key,
                "confidence": 1.0, "reason": "already a valid catalog code"}

    cached = _cache_get("job_title", job_title)
    if cached:
        cached["reason"] = "[cache] " + cached.get("reason", "")
        return cached

    catalog_lines = "\n".join(f"{c} -- {d}" for c, d in JOB_CATALOG.items())
    prompt = f"""You map messy HRIS job titles to Acme Health's Workday job catalog.
Reply with ONLY valid JSON: {{"workday_value": "...", "confidence": 0.0, "reason": "..."}}

Catalog (code -- description):
{catalog_lines}

Examples:
"RN"               -> {{"workday_value":"RN","confidence":0.99,"reason":"abbrev"}}
"Registered Nurse" -> {{"workday_value":"RN","confidence":0.99,"reason":"exact"}}
"Nurse Practitioner" -> {{"workday_value":"NP_FAM","confidence":0.92,"reason":"family default for Acme Health NPs"}}

Input: "{job_title}"
"""
    result = _call_ollama(prompt)
    _cache_set("job_title", job_title, result)
    return result


@lru_cache(maxsize=128)
def normalize_department(dept_name: str) -> dict:
    key = dept_name.strip().upper()

    if key in DEPT_SHORTHAND:
        return {"workday_value": DEPT_SHORTHAND[key],
                "confidence": 1.0, "reason": "exact shorthand match"}

    cached = _cache_get("department", dept_name)
    if cached:
        cached["reason"] = "[cache] " + cached.get("reason", "")
        return cached

    allowed_str = ", ".join(f'"{d}"' for d in ALLOWED_DEPARTMENTS)
    prompt = f"""You normalize HRIS department names for Acme Health's Workday migration.
Reply with ONLY valid JSON: {{"workday_value": "...", "confidence": 0.0, "reason": "..."}}

Allowed output values: [{allowed_str}]
Use "Unknown" only if nothing fits.

Input: "{dept_name}"
"""
    result = _call_ollama(prompt)
    _cache_set("department", dept_name, result)
    return result


@lru_cache(maxsize=64)
def normalize_location(location_name: str) -> dict:
    key = location_name.strip().upper()

    if key in LOCATION_SHORTHAND:
        return {"workday_value": LOCATION_SHORTHAND[key],
                "confidence": 1.0, "reason": "exact shorthand match"}
    if key in LOCATION_CODES:
        return {"workday_value": key,
                "confidence": 1.0, "reason": "already a valid location code"}

    cached = _cache_get("location", location_name)
    if cached:
        cached["reason"] = "[cache] " + cached.get("reason", "")
        return cached

    loc_lines = "\n".join(f"{c} -- {city}" for c, city in LOCATION_CODES.items())
    prompt = f"""You map clinic location names to Workday location codes for Acme Health.
Reply with ONLY valid JSON: {{"workday_value": "...", "confidence": 0.0, "reason": "..."}}

Location codes (code -- city):
{loc_lines}

Examples:
"Searcy AR"  -> {{"workday_value":"SRCY","confidence":0.97,"reason":"state suffix"}}
"Remote"     -> {{"workday_value":"REM","confidence":0.99,"reason":"telehealth"}}

Input: "{location_name}"
"""
    result = _call_ollama(prompt)
    _cache_set("location", location_name, result)
    return result


# =========================================================
# DATAFRAME MAPPER  --  UNIQUE VALUE MAPPING
# needs_human_review is DEFINED HERE, not in cleaning_real.py.
# =========================================================

def apply_llm_mappings(df):
    """
    Maps Job_Title, Department_Name, Location_Name using unique-value mapping.

    Unique value mapping:
      - Extract unique values per column (e.g. 8 unique titles from 62 rows)
      - Call mapper once per unique value
      - Join results back with pandas .map() (vectorised, no row loop)

    Persistent cache:
      - Before calling Ollama, checks mapping_cache table in employees.db
      - After calling Ollama, writes result to mapping_cache
      - On next pipeline run, previously seen values never hit Ollama

    needs_human_review:
      - Defined here, after LLM mapping produces confidence scores
      - Never exists on a DataFrame that hasn't been through this function
      - etl_pipeline.py reads it only after this function returns

    Adds columns:
      workday_job_code, job_title_confidence, job_title_reason
      workday_department, department_confidence, department_reason
      workday_location_code, location_confidence, location_reason
      llm_confidence           -- min of three (field-mapping doc audit field)
      needs_human_review       -- True if any confidence < 0.8
      ai_transformation_reason -- composite reason string
      review_status            -- "auto" or "review"
    """
    _ensure_cache_table()

    def map_unique(column: str, mapper_fn):
        unique_vals = df[column].dropna().unique()
        results     = {val: mapper_fn(str(val)) for val in unique_vals}
        return (
            {v: r.get("workday_value") for v, r in results.items()},
            {v: r.get("confidence", 0.0) for v, r in results.items()},
            {v: r.get("reason", "")     for v, r in results.items()},
        )

    # --- Job title ---
    jv, jc, jr = map_unique("Job_Title", normalize_job_title)
    df["workday_job_code"]     = df["Job_Title"].map(jv)
    df["job_title_confidence"] = df["Job_Title"].map(jc)
    df["job_title_reason"]     = df["Job_Title"].map(jr)

    # --- Department ---
    dv, dc, dr = map_unique("Department_Name", normalize_department)
    df["workday_department"]    = df["Department_Name"].map(dv)
    df["department_confidence"] = df["Department_Name"].map(dc)
    df["department_reason"]     = df["Department_Name"].map(dr)

    # --- Location ---
    lv, lc, lr = map_unique("Location_Name", normalize_location)
    df["workday_location_code"] = df["Location_Name"].map(lv)
    df["location_confidence"]   = df["Location_Name"].map(lc)
    df["location_reason"]       = df["Location_Name"].map(lr)

    # --- Derived audit fields ---
    df["llm_confidence"] = df[[
        "job_title_confidence", "department_confidence", "location_confidence"
    ]].min(axis=1)

    # needs_human_review: defined here, only here
    df["needs_human_review"] = (
        df["llm_confidence"].fillna(0) < CONFIDENCE_THRESHOLD
    )

    df["ai_transformation_reason"] = (
        "Job: "  + df["job_title_reason"].fillna("n/a")  + " | " +
        "Dept: " + df["department_reason"].fillna("n/a") + " | " +
        "Loc: "  + df["location_reason"].fillna("n/a")
    )

    df["review_status"] = df["needs_human_review"].map(
        {True: "review", False: "auto"}
    )

    return df


# =========================================================
# SMOKE TEST
# =========================================================

if __name__ == "__main__":
    _ensure_cache_table()
    cases = [
        ("JOB",  normalize_job_title,  [
            "RN", "Registered Nurse", "Nurse RN",
            "Front Desk Coordinator", "Family Physician MD",
        ]),
        ("DEPT", normalize_department, [
            "HR", "Human Resources", "People Ops", "Nursing",
        ]),
        ("LOC",  normalize_location,   [
            "Searcy", "Searcy AR", "Newport", "Remote",
        ]),
    ]
    for label, fn, inputs in cases:
        print(f"\n===== {label} =====")
        for val in inputs:
            r = fn(val)
            flag = " ⚠ HUMAN" if r.get("confidence", 1) < CONFIDENCE_THRESHOLD else ""
            print(f"  {val!r:35s} -> {r.get('workday_value')} "
                  f"({r.get('confidence', 0):.2f}){flag}")