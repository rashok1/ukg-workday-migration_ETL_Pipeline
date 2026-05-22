# UKG -> Workday Field Mapping (Data Dictionary)

**Scope:** Employee master record for the Acme Health migration.
**Format:** Source column -> Workday target -> transformation -> who owns it.
**Legend:** RULE = deterministic Pandas rule. LLM = private LLM (Ollama)
fuzzy mapping. HUMAN = manual review before load.

## Core demographic fields

| UKG column        | Workday target           | Transformation                                          | Owner |
|-------------------|--------------------------|---------------------------------------------------------|-------|
| Employee_ID       | External_ID              | Keep as `source_record_id`; Workday assigns its own.    | RULE  |
| First_Name        | First_Name               | Trim, title-case, strip non-printables.                 | RULE  |
| Last_Name         | Last_Name                | Trim, title-case, strip non-printables.                 | RULE  |
| Birth_Date        | Date_of_Birth            | Parse mixed formats, ISO-format the result.             | RULE  |
| Gender            | Gender                   | Normalize (M/F -> Male/Female, blank -> Not Specified). | RULE  |
| Email             | Work_Email               | Lowercase; validate; flag missing.                      | RULE  |

## Job and org fields (LLM-heavy)

| UKG column        | Workday target              | Transformation                                                       | Owner |
|-------------------|-----------------------------|----------------------------------------------------------------------|-------|
| Job_Title         | Job_Profile (catalog code)  | LLM maps free text -> Workday job catalog code w/ confidence + reason.| LLM   |
| Department_Name   | Supervisory_Org / Cost_Ctr  | LLM maps to Acme Health's supervisory-org tree.                            | LLM   |
| Location_Name     | Workday Location code       | Strip "HQ"/"Office", LLM maps city -> location code (e.g. Searcy -> SRCY). | LLM |
| Manager_ID        | Manager_External_ID         | Lookup against staged data; gap-fill via LLM proposal -> HUMAN sign-off. | LLM + HUMAN |
| Pay_Type          | Pay_Rate_Type               | Map Salary -> Salary, Hourly -> Hourly; blank -> review.              | RULE  |

## Compensation

| UKG column        | Workday target           | Transformation                                              | Owner |
|-------------------|--------------------------|-------------------------------------------------------------|-------|
| Annual_Salary     | Annual_Base_Pay          | Coerce to numeric; flag "invalid" -> exceptions queue.      | RULE  |
| Annual_Salary     | Pay_Range_Band_Check     | Sanity-check against job-profile band; out-of-band -> HUMAN.| HUMAN |
| Hire_Date         | Hire_Date                | Parse mixed formats; "invalid" -> exceptions queue.         | RULE  |

## Audit fields (added during ETL, not in UKG)

| Workday target / staging col       | Source             | Purpose                                           |
|------------------------------------|--------------------|---------------------------------------------------|
| source_system                      | constant `UKG`     | Lineage.                                          |
| source_record_id                   | Employee_ID        | Reverse lookup if Workday assigns a new key.      |
| load_timestamp                     | clock              | When ETL ran.                                     |
| ai_transformation_reason           | LLM JSON `reason`  | Per-field rationale for auditors.                 |
| llm_confidence                     | LLM JSON           | Numeric, 0.0-1.0; below 0.8 routes to HUMAN.      |
| review_status                      | reviewer action    | `auto`, `approved`, `overridden`.                 |

## Exception categories (rows that don't make it through clean)

| Category                  | Signal                                | Routing              |
|---------------------------|---------------------------------------|----------------------|
| Bad date                  | NaT after parse                       | Streamlit Data Quality tab |
| Bad salary                | NaN after numeric coerce              | Streamlit Data Quality tab |
| Missing dept              | Department_Name = `Unknown`           | LLM mapping retry; else HUMAN |
| Duplicate                 | Same (First, Last, DOB)               | Drop, log to `dupes` table |
| Manager not found         | Manager_ID has no match               | LLM proposal; HUMAN sign-off |
| Out-of-band salary        | Outside job-profile band              | HUMAN |

## Job catalog seed (starter -- expand with Acme Health's actual catalog)

```
PHYS_FAM       Family Physician (MD/DO)
PHYS_INTERN    Internal Medicine Physician
NP_FAM         Nurse Practitioner -- Family
RN             Registered Nurse
LPN            Licensed Practical Nurse
MA             Medical Assistant
PHARM_RX       Pharmacist
PHARM_TECH     Pharmacy Technician
DENT_DDS       Dentist
DENT_HYG       Dental Hygienist
BH_LCSW        Behavioral Health Counselor (LCSW)
ADMIN_FD       Front Desk / Patient Coordinator
ADMIN_PM       Practice Manager
ADMIN_CMO      Chief Medical Officer
BILL_SPEC      Billing Specialist
BILL_CODER     Medical Coder
IT_SUP         IT Support
HR_SPEC        HR Specialist
FIN_ANALYST    Finance Analyst
OPS_MGR        Operations Manager
```

## Location seed (real Acme Health clinic cities)

```
AUG    Augusta
BAT    Batesville
SRCY   Searcy
HBR    Heber Springs
MTN    Mountain View
NWP    Newport
CBT    Cabot
BEE    Beebe
CLN    Clinton
CRN    Corning
REM    Remote / Telehealth
```

## Sample LLM mapping prompt (job title)

```
You map messy HRIS job titles to Acme Health's Workday job catalog.
Reply with ONLY valid JSON: {"workday_code": "...", "confidence": 0.0-1.0, "reason": "..."}

Catalog (code -- description):
PHYS_FAM -- Family Physician (MD/DO)
NP_FAM   -- Nurse Practitioner -- Family
RN       -- Registered Nurse
LPN      -- Licensed Practical Nurse
MA       -- Medical Assistant
...

Examples:
"RN"               -> {"workday_code":"RN","confidence":0.99,"reason":"abbrev"}
"Registered Nurse" -> {"workday_code":"RN","confidence":0.99,"reason":"exact"}
"Nurse Practitioner" -> {"workday_code":"NP_FAM","confidence":0.92,"reason":"family is default specialty for Acme Health NPs"}

Input: "{{job_title}}"
```

## Where deterministic stops and LLM starts

Use the **two-list test**:

- If the source values are drawn from a known finite vocabulary that
  you could maintain in a YAML file -> RULE.
- If the source values are free text written by humans on different
  days with different conventions -> LLM.

`Pay_Type` is rules. `Job_Title` is LLM. `Department_Name` is mostly
LLM but with a strong prior from the rules table.
