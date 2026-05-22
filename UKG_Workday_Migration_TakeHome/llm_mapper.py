"""
llm_mapper.py
-------------
Stage 4 of the pipeline: LLM-based fuzzy mapping.

Per the field-mapping doc, LLM is ONLY used for these three fields:
  - Job_Title      -> Workday job catalog CODE  (e.g. "RN", "PHYS_FAM")
  - Department_Name -> Workday supervisory org label
  - Location_Name   -> Workday location CODE    (e.g. "SRCY")

All other fields (dates, salary, gender, email, pay type) are RULE-owned
and stay in cleaning_real.py. This module never touches them.

Caching strategy (per your question):
  1. Check hardcoded lookup dict first -- known values return instantly
     at confidence 1.0, no LLM call at all.
  2. On a miss, call LLM once and cache the result with @lru_cache.
     So "Registered Nurse" hits the model once; the 29 other rows
     with the same value get the cached result.

JSON safety:
  llama3 wraps output in ```json ... ``` fences even when told not to.
  We strip those before json.loads() -- without this every call silently
  fails and returns confidence 0.0.

No PHI in prompts (process flow doc Stage 4):
  Only the fuzzy field value is sent. Name, DOB, salary stay in SQLite.

Audit trail (field-mapping doc):
  Every mapping returns {"workday_value", "confidence", "reason"}.
  confidence < 0.8 -> needs_human_review = True -> Streamlit review queue.
  The reason field populates ai_transformation_reason in the DB.
"""

import re
import json
import ollama
from functools import lru_cache


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "llama3:latest"
CONFIDENCE_THRESHOLD = 0.8  # below this -> routed to human review


# =========================================================
# CATALOGS  (source: UKG_to_Workday_Field_Mapping.md)
# =========================================================

# The authoritative job catalog from the spec.
# Keys are Workday codes; values are human-readable descriptions.
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

# Hardcoded shorthand -> code. These are unambiguous; no LLM needed.
# Extend this as you discover repeated patterns in the raw data.
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

# Location codes from the spec.
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

# Reverse lookup: cleaned city name -> code.
# cleaning_real.py already strips "HQ"/"Office", so inputs here are
# already clean city names. These exact matches need no LLM.
LOCATION_SHORTHAND = {v.upper(): k for k, v in LOCATION_CODES.items()}
LOCATION_SHORTHAND["REMOTE"] = "REM"
LOCATION_SHORTHAND["TELEHEALTH"] = "REM"

# Department: known variants -> canonical label.
# Field-mapping doc says "mostly LLM but with a strong prior from the
# rules table" -- this is that rules table.
DEPT_SHORTHAND = {
    "HR":                   "Human Resources",
    "HUMAN RESOURCES":      "Human Resources",
    "PEOPLE":               "Human Resources",
    "IT":                   "IT",
    "INFORMATION TECHNOLOGY": "IT",
    "FINANCE":              "Finance",
    "BILLING":              "Billing",
    "ADMIN":                "Administration",
    "ADMINISTRATION":       "Administration",
    "OPERATIONS":           "Operations",
    "OPS":                  "Operations",
    "PHARMACY":             "Pharmacy",
    "PRIMARY CARE":         "Primary Care",
    "NURSING":              "Primary Care",
    "DENTAL":               "Dental",
    "BEHAVIORAL HEALTH":    "Behavioral Health",
    "BH":                   "Behavioral Health",
    "UNKNOWN":              "Unknown",
}

ALLOWED_DEPARTMENTS = sorted(set(DEPT_SHORTHAND.values()))


# =========================================================
# CORE LLM CALL
# =========================================================

def _call_ollama(prompt: str) -> dict:
    """
    Sends prompt to local Ollama. Strips markdown fences before
    parsing -- llama3 adds them even when told not to, which would
    cause json.loads to throw and silently return confidence 0.0
    on every single call.
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"].strip()
        # Strip ```json ... ``` or ``` ... ``` fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "workday_value": None,
            "confidence": 0.0,
            "reason": "model returned non-JSON output",
        }
    except Exception as e:
        return {
            "workday_value": None,
            "confidence": 0.0,
            "reason": f"LLM error: {e}",
        }


# =========================================================
# JOB TITLE MAPPER
# =========================================================

@lru_cache(maxsize=256)
def normalize_job_title(job_title: str) -> dict:
    """
    Maps free-text Job_Title -> Workday catalog CODE.

    Order of resolution:
      1. Hardcoded JOB_SHORTHAND -> confidence 1.0, no LLM call.
      2. Input already is a valid catalog code -> confidence 1.0.
      3. LLM with few-shot prompt (spec sample prompt) -> cached.
    """
    key = job_title.strip().upper()

    if key in JOB_SHORTHAND:
        return {
            "workday_value": JOB_SHORTHAND[key],
            "confidence": 1.0,
            "reason": "exact shorthand match -- no LLM needed",
        }

    if key in JOB_CATALOG:
        return {
            "workday_value": key,
            "confidence": 1.0,
            "reason": "input is already a valid catalog code",
        }

    # LLM path -- prompt pattern taken directly from the spec
    catalog_lines = "\n".join(
        f"{code} -- {desc}" for code, desc in JOB_CATALOG.items()
    )
    prompt = f"""You map messy HRIS job titles to Acme Health's Workday job catalog.
Reply with ONLY valid JSON: {{"workday_value": "...", "confidence": 0.0, "reason": "..."}}

Catalog (code -- description):
{catalog_lines}

Examples:
"RN"               -> {{"workday_value":"RN","confidence":0.99,"reason":"abbrev"}}
"Registered Nurse" -> {{"workday_value":"RN","confidence":0.99,"reason":"exact"}}
"Nurse Practitioner" -> {{"workday_value":"NP_FAM","confidence":0.92,"reason":"family is default specialty for Acme Health NPs"}}

Input: "{job_title}"
"""
    return _call_ollama(prompt)


# =========================================================
# DEPARTMENT MAPPER
# =========================================================

@lru_cache(maxsize=128)
def normalize_department(dept_name: str) -> dict:
    """
    Maps Department_Name -> Workday supervisory org label.

    Order of resolution:
      1. Hardcoded DEPT_SHORTHAND -> confidence 1.0, no LLM.
      2. LLM with constrained output list -> cached.
    """
    key = dept_name.strip().upper()

    if key in DEPT_SHORTHAND:
        return {
            "workday_value": DEPT_SHORTHAND[key],
            "confidence": 1.0,
            "reason": "exact shorthand match -- no LLM needed",
        }

    allowed_str = ", ".join(f'"{d}"' for d in ALLOWED_DEPARTMENTS)
    prompt = f"""You normalize HRIS department names for Acme Health's Workday migration.
Reply with ONLY valid JSON: {{"workday_value": "...", "confidence": 0.0, "reason": "..."}}

Allowed output values: [{allowed_str}]
Use "Unknown" only if nothing fits.

Input: "{dept_name}"
"""
    return _call_ollama(prompt)


# =========================================================
# LOCATION MAPPER
# =========================================================

@lru_cache(maxsize=64)
def normalize_location(location_name: str) -> dict:
    """
    Maps Location_Name -> Workday location CODE (e.g. "SRCY").

    Note: cleaning_real.py strips "HQ"/"Office" before this runs,
    so input here is already a plain city name.

    Order of resolution:
      1. Hardcoded LOCATION_SHORTHAND -> confidence 1.0, no LLM.
      2. Input already is a valid code -> confidence 1.0.
      3. LLM -> cached.
    """
    key = location_name.strip().upper()

    if key in LOCATION_SHORTHAND:
        return {
            "workday_value": LOCATION_SHORTHAND[key],
            "confidence": 1.0,
            "reason": "exact shorthand match -- no LLM needed",
        }

    if key in LOCATION_CODES:
        return {
            "workday_value": key,
            "confidence": 1.0,
            "reason": "input is already a valid location code",
        }

    loc_lines = "\n".join(
        f"{code} -- {city}" for code, city in LOCATION_CODES.items()
    )
    prompt = f"""You map clinic location names to Workday location codes for Acme Health.
Reply with ONLY valid JSON: {{"workday_value": "...", "confidence": 0.0, "reason": "..."}}

Location codes (code -- city):
{loc_lines}

Examples:
"Searcy"     -> {{"workday_value":"SRCY","confidence":0.99,"reason":"exact city match"}}
"Searcy AR"  -> {{"workday_value":"SRCY","confidence":0.97,"reason":"state suffix stripped"}}
"Remote"     -> {{"workday_value":"REM","confidence":0.99,"reason":"telehealth/remote"}}

Input: "{location_name}"
"""
    return _call_ollama(prompt)


# =========================================================
# DATAFRAME MAPPER
# =========================================================

def apply_llm_mappings(df):
    """
    Applies all three mappers to the dataframe.

    Adds these columns (matching field-mapping doc audit fields):
      workday_job_code, job_title_confidence, job_title_reason
      workday_department, department_confidence, department_reason
      workday_location_code, location_confidence, location_reason
      llm_confidence       -- lowest of the three (single audit field per spec)
      needs_human_review   -- True if any confidence < 0.8
      ai_transformation_reason -- composite reason string for auditors
      review_status        -- "auto" or "review" per field-mapping doc
    """
    def unpack(series, key):
        return series.apply(
            lambda x: x.get(key) if isinstance(x, dict) else None
        )

    job_results  = df["Job_Title"].apply(normalize_job_title)
    dept_results = df["Department_Name"].apply(normalize_department)
    loc_results  = df["Location_Name"].apply(normalize_location)

    df["workday_job_code"]      = unpack(job_results,  "workday_value")
    df["job_title_confidence"]  = unpack(job_results,  "confidence")
    df["job_title_reason"]      = unpack(job_results,  "reason")

    df["workday_department"]    = unpack(dept_results, "workday_value")
    df["department_confidence"] = unpack(dept_results, "confidence")
    df["department_reason"]     = unpack(dept_results, "reason")

    df["workday_location_code"] = unpack(loc_results,  "workday_value")
    df["location_confidence"]   = unpack(loc_results,  "confidence")
    df["location_reason"]       = unpack(loc_results,  "reason")

    # llm_confidence: single summary field per the field-mapping doc audit table
    df["llm_confidence"] = df[[
        "job_title_confidence",
        "department_confidence",
        "location_confidence",
    ]].min(axis=1)

    # needs_human_review: any field below threshold
    df["needs_human_review"] = df["llm_confidence"].fillna(0) < CONFIDENCE_THRESHOLD

    # ai_transformation_reason: composite string for auditors
    df["ai_transformation_reason"] = (
        "Job: "  + df["job_title_reason"].fillna("n/a")   + " | " +
        "Dept: " + df["department_reason"].fillna("n/a")  + " | " +
        "Loc: "  + df["location_reason"].fillna("n/a")
    )

    # review_status per field-mapping doc: "auto" or "review"
    df["review_status"] = df["needs_human_review"].map(
        {True: "review", False: "auto"}
    )

    return df


# =========================================================
# SMOKE TEST
# =========================================================

if __name__ == "__main__":
    cases = [
        ("JOB",  normalize_job_title,  [
            "RN", "Registered Nurse", "Nurse RN",
            "Front Desk Coordinator", "Family Physician MD",
        ]),
        ("DEPT", normalize_department, [
            "HR", "Human Resources", "People Ops", "Nursing",
        ]),
        ("LOC",  normalize_location,   [
            "Searcy", "Searcy AR", "Newport Office", "Remote",
        ]),
    ]
    for label, fn, inputs in cases:
        print(f"\n===== {label} =====")
        for val in inputs:
            r = fn(val)
            flag = " ⚠ HUMAN" if r.get("confidence", 1) < CONFIDENCE_THRESHOLD else ""
            print(f"  {val!r:35s} -> {r.get('workday_value')} "
                  f"({r.get('confidence', 0):.2f}){flag}")