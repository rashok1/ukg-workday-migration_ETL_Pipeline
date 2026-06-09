# Acme Health UKG -> Workday Migration Take-Home

Privacy-first ETL with a **local LLM (Ollama)** + SQLite + Streamlit.
All data and inference stay on your machine -- nothing is sent to a cloud LLM.

## Context

Acme Health is a community health center operating across Arkansas (Augusta,
Batesville, Searcy, Heber Springs, Mountain View, Newport, Cabot, Beebe,
Clinton, Corning). We're migrating employee records from **UKG** to **Workday**.
Because the export contains PHI-adjacent data (names, DOB, salaries, work
locations tied to clinics), we run inference locally.

## What's in this ZIP

```
UKG_Workday_Migration_TakeHome/
  ukg_employees_raw.csv      # Messy UKG employee export (80 rows)
  requirements.txt           # Python deps
  etl_pipeline.py            # Starter ETL (extend with Ollama mapping)
  app.py                     # Streamlit dashboard
  README.md                  # This file
```

## Setup

1. `pip install -r requirements.txt`
2. Install Ollama (https://ollama.com) and pull a model:
   `ollama pull qwen2.5:7b` (or `llama3.1:8b`)
3. `python etl_pipeline.py`     -- builds `employees.db`
4. `streamlit run app.py`       -- opens the dashboard

## What we want you to build

The starter scripts are intentionally minimal. Extend them to demonstrate:

1. **Intelligent mapping via local LLM** -- use LangChain + Ollama to map
   messy `Job_Title`, `Department_Name`, and `Location_Name` values onto a
   standardized Workday catalog. Log which rows the LLM touched in the
   `ai_transformation_reason` column.
2. **Stronger cleaning** -- deduplication, salary band sanity checks,
   email format validation, manager-ID referential integrity.
3. **AI Insights tab** -- wire the dashboard's "Get Insight" button to an
   actual Ollama call against the cleaned dataset.
4. **Production notes** -- briefly cover how you'd schedule this (Airflow,
   cron), how you'd handle PHI (HIPAA considerations, local-only inference),
   and how you'd test it (unit tests, data quality assertions).

## Submission

GitHub repo or zip file by the deadline in the email.
Include a short Loom or written walkthrough of your design decisions.
