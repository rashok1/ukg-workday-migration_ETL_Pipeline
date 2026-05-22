# Acme Health Private LLM + UKG -> Workday Migration -- Process Flow

**Owner:** Data Migration Team, Acme Health
**Status:** Draft v1.0
**Last updated:** May 2026

## 1. Why a private LLM

Acme Health is a federally qualified community health center. Employee records
contain PHI-adjacent data (DOB, work clinic, salary band), and clinical
context that touches the LLM later -- chart notes, encounter summaries,
prior-auth letters -- is squarely PHI. Sending that data to a public
LLM API is a HIPAA non-starter and a contracting headache (BAA, data
residency, retention).

A **private LLM** means:

- The model weights live on infrastructure Acme Health controls (on-prem GPU
  box or a single-tenant VPC).
- Inference, prompts, and retrieved documents never leave that boundary.
- We can audit every prompt + response for the HIPAA Security Rule.

The UKG -> Workday migration is the first real workload, but the
architecture is reusable for downstream use cases (clinical Q&A over
policy docs, billing code lookup, intake summarization).

## 2. End-to-end flow

```
+-----------+    +-------------+    +------------+    +-------------+    +-------------+
|  UKG      |    |  ETL        |    |  SQLite    |    |  Private    |    |  Workday    |
|  Export   |--->|  pipeline   |--->|  staging   |--->|  LLM        |--->|  EIB / API  |
|  (CSV)    |    |  (Pandas)   |    |  warehouse |    |  (Ollama)   |    |  load       |
+-----------+    +-------------+    +------------+    +-------------+    +-------------+
      |                |                  |                  |                  |
      v                v                  v                  v                  v
  raw, messy       deterministic       audit-friendly     fuzzy /          validated
  multi-format     cleaning            queryable          semantic         Workday rows
                   (dates, $, nulls)   intermediate       mapping          with audit
                                                          (titles, depts,  trail
                                                           locations)

           +-------------------------------------------------+
           |  Streamlit dashboard (read-only review surface) |
           |  - QA the cleaned rows                          |
           |  - Inspect LLM mapping decisions                |
           |  - Ask natural-language questions               |
           +-------------------------------------------------+
```

## 3. Stage-by-stage detail

### Stage 1 -- Extract (UKG)
- Pull employee export from UKG via the scheduled report job (CSV).
- Drop the file into a watched directory (`/data/landing/ukg/`).
- File-naming convention: `ukg_employees_YYYYMMDD.csv`.
- Hash the file and log it to `audit.ingest_log`.

### Stage 2 -- Deterministic cleaning (Pandas)
What the rule engine handles -- no LLM needed, fully reproducible:

| Issue                              | Rule                                                   |
|------------------------------------|--------------------------------------------------------|
| Mixed date formats                 | `pd.to_datetime(..., errors='coerce')`, fall back to `dateutil`. |
| Free-text numbers in salary column | `pd.to_numeric(..., errors='coerce')`, flag NaN.       |
| Missing department/location/email  | Fill with sentinel + add to exception report.          |
| "M"/"F" gender codes               | Map to `Male`/`Female`/`Not Specified`.                |
| Trailing "HQ"/"Office" on location | Strip via regex.                                       |
| Duplicate rows                     | Drop on `(First_Name, Last_Name, Birth_Date)`.         |

### Stage 3 -- SQLite staging
- Table: `workday_employees` -- the cleaned mid-point.
- Every row carries audit columns:
  - `source_system` (always `UKG` for this load)
  - `source_record_id` (the UKG Employee_ID)
  - `load_timestamp`
  - `ai_transformation_reason` (free text, populated by Stage 4)
- Separate `exceptions` table for rows that failed validation.

### Stage 4 -- Private LLM mapping (Ollama)
The LLM is invoked **only** for fuzzy/judgment tasks where rules don't
generalize cleanly:

| Source value              | Target (Workday)            | Why LLM                                |
|---------------------------|-----------------------------|----------------------------------------|
| `Job_Title` free text     | Workday job catalog code    | "RN", "Registered Nurse", "Nurse RN"   |
|                           |                             | all -> same code; rules brittle.       |
| `Department_Name`         | Workday Supervisory Org     | "HR" vs "Human Resources" vs "People"; |
|                           |                             | maps to a finite vocabulary.           |
| `Location_Name`           | Workday Location code       | "Searcy HQ", "Searcy AR", "Searcy"     |
|                           |                             | -> SRCY clinic.                        |
| Manager_ID nulls (later)  | Inferred from org chart     | LLM proposes candidate; human signs.   |

**Prompt pattern:** few-shot, constrained-output. The model is asked to
return JSON only, with `{ "workday_value": "...", "confidence": 0.0-1.0,
"reason": "..." }`. Anything below `confidence >= 0.8` is routed to a
human reviewer in the Streamlit dashboard.

**Model choice (default):** `qwen2.5:7b-instruct` or `llama3.1:8b-instruct`
via Ollama. Both fit comfortably on a single 24 GB GPU (RTX A5000-class)
or even on Apple Silicon for the dev environment. See the architecture
recommendation doc for sizing.

**No PHI in prompts.** For Stage 4 mapping we only send the columns
needed for the mapping decision (e.g. `Job_Title` + sample of similar
titles). Name, DOB, salary stay in SQLite.

### Stage 5 -- Workday load
- Generate Workday EIB (Excel template) or call the Workday SOAP/REST
  API depending on the object.
- Use the Workday-Journal-EIB skill for finance-adjacent loads.
- Run validation against the Workday job catalog before submitting.
- Every load produces a `workday_load_log` row with the submission ID.

### Stage 6 -- Review surface (Streamlit)
- Tabs: Overview, Data Explorer, Data Quality, AI Insights.
- Reviewers approve/override LLM mappings.
- Overrides write back to a `mapping_overrides` table that feeds the
  next ETL run (compounding accuracy).

## 4. Sequence diagram (happy path, single record)

```
UKG file        ETL              SQLite          Ollama          Workday
   |              |                 |                |               |
   |--push------->|                 |                |               |
   |              |--clean----------|                |               |
   |              |--upsert row---->|                |               |
   |              |                 |--needs map?--->|               |
   |              |                 |<--{value,0.94} |               |
   |              |--update row---->|                |               |
   |              |                 |---reviewed---->| (Streamlit)   |
   |              |                 |                |               |
   |              |--export EIB-------------------------------------->|
   |              |                 |                |               |
   |              |<------------------------submission ID-------------|
```

## 5. Failure modes and the answer to each

| Failure                                     | How we catch it                              | Recovery                                |
|---------------------------------------------|----------------------------------------------|-----------------------------------------|
| UKG export schema drifts (new/renamed col)  | Schema check at Stage 1                      | Hard-fail load, page on-call.           |
| Ollama is down                              | Health check before Stage 4                  | Queue rows; resume when service returns.|
| LLM low-confidence mapping                  | `confidence < 0.8` threshold                 | Route to human review tab in Streamlit. |
| Duplicate Workday record                    | Pre-load lookup by source_record_id          | Update instead of insert; log delta.    |
| PHI leak risk in prompt                     | Prompt template lint -- allow-list of fields | Block send, alert security.             |

## 6. What changes when this scales beyond migration

The same pipeline is the foundation for Acme Health's broader private LLM
ambitions:

- **RAG over policy + clinical content** -- swap SQLite staging for a
  vector store (Chroma or FAISS), keep Ollama as the inference layer.
- **Agentic workflows** -- a LangChain agent over the same models can
  draft prior-auth letters or summarize encounter notes, with the
  audit table extended to record tool calls.
- **Model upgrades** -- because the model layer is just an Ollama
  endpoint, swapping `qwen2.5:7b` for a domain-tuned model (or a
  larger 14B/32B variant on bigger hardware) is a config change.

## 7. Open questions for the team

1. Where does the GPU live -- on-prem at Searcy HQ, or a single-tenant
   VPC at a cloud provider that will sign a BAA?
2. Who owns the Workday job catalog and how often is it updated?
3. Do we need a redaction layer before any prompt leaves the SQLite
   boundary, even for non-PHI fields?
4. Retention policy for `ai_transformation_reason` -- the prompts and
   model responses themselves count as PHI when they describe employees.
