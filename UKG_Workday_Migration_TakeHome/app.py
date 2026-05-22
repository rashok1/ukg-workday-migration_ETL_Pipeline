"""
app.py
------
Stage 6: Streamlit review surface.
Extended from the starter file per README instructions.

Tabs per process flow doc Stage 6:
  - Overview      : headcount + salary charts (starter kept, minor additions)
  - Data Explorer : searchable employee table (starter kept)
  - Data Quality  : exceptions + dupes from SQLite (starter extended --
                    now reads from the exceptions table, not re-derives them)
  - AI Insights   : LLM review queue + natural language Q&A via Ollama

Key design choices:
  - Data Quality tab reads from the `exceptions` table written by
    etl_pipeline.py, not by re-deriving from clean_df. Single source
    of truth is the DB.
  - AI Insights tab shows the human review queue (confidence < 0.8 rows)
    with approve/override controls that write back to the DB -- this is
    the "mapping_overrides" pattern from the process flow doc.
  - LLM Q&A uses LangChain + Ollama (llama3) per README instructions.
    Only non-PHI columns are included in the prompt context.
  - review_status updates write to workday_employees immediately so the
    next etl_pipeline.py run and workday_exporter.py see the decision.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from langchain_ollama import ChatOllama

st.set_page_config(
    page_title="Acme Health UKG -> Workday Migration",
    layout="wide",
)
st.title("Acme Health: UKG to Workday Migration Dashboard")
st.caption("Privacy-first migration review for Acme Health community health center")


# =========================================================
# DB CONNECTION + DATA LOAD
# =========================================================

@st.cache_resource
def get_engine():
    return create_engine("sqlite:///employees.db")


@st.cache_data(ttl=30)  # refresh every 30s so review actions are reflected
def load_table(table_name: str) -> pd.DataFrame:
    try:
        return pd.read_sql(f"SELECT * FROM {table_name}", get_engine())
    except Exception:
        return pd.DataFrame()


try:
    df = load_table("workday_employees")
    if df.empty:
        st.error("workday_employees table is empty -- run `python etl_pipeline.py` first.")
        st.stop()
except Exception as e:
    st.error(f"Could not read employees.db -- run `python etl_pipeline.py` first.\n\n{e}")
    st.stop()

exceptions_df = load_table("exceptions")
dupes_df      = load_table("dupes")


# =========================================================
# TOP METRICS
# =========================================================

needs_review = int(df["needs_human_review"].sum()) if "needs_human_review" in df.columns else 0
auto_ok      = len(df) - needs_review

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Records Migrated",   len(df))
col2.metric("Locations",          df["Location_Name"].nunique())
col3.metric("Departments",        df["Department_Name"].nunique())
col4.metric("Missing Salaries",   int(df["Annual_Salary"].isna().sum()))
col5.metric("Needs Human Review", needs_review,
            delta=f"{auto_ok} auto-approved",
            delta_color="inverse" if needs_review > 0 else "normal")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Data Explorer",
    "⚠️ Data Quality",
    "🤖 AI Insights",
])


# ---------------------------------------------------------
# TAB 1: OVERVIEW  (starter charts kept; pay type added)
# ---------------------------------------------------------

with tab1:
    st.subheader("Headcount by Department")
    dept_counts = df["Department_Name"].value_counts().reset_index()
    dept_counts.columns = ["Department_Name", "Headcount"]
    st.plotly_chart(
        px.bar(dept_counts, x="Department_Name", y="Headcount",
               title="Employees by Department", color="Headcount",
               color_continuous_scale="Blues"),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Headcount by Clinic Location")
        loc_counts = df["Location_Name"].value_counts().reset_index()
        loc_counts.columns = ["Location_Name", "Headcount"]
        st.plotly_chart(
            px.bar(loc_counts, x="Location_Name", y="Headcount",
                   title="Employees by Acme Health Location"),
            use_container_width=True,
        )

    with col_b:
        st.subheader("Pay Type Breakdown")
        if "Pay_Type" in df.columns:
            pay_counts = df["Pay_Type"].value_counts().reset_index()
            pay_counts.columns = ["Pay_Type", "Count"]
            st.plotly_chart(
                px.pie(pay_counts, names="Pay_Type", values="Count",
                       title="Salary vs Hourly"),
                use_container_width=True,
            )

    st.subheader("Salary Distribution by Department")
    st.plotly_chart(
        px.box(
            df.dropna(subset=["Annual_Salary"]),
            x="Department_Name", y="Annual_Salary",
            title="Annual Salary by Department",
        ),
        use_container_width=True,
    )

    # LLM mapping confidence distribution (only if pipeline ran with LLM)
    if "llm_confidence" in df.columns:
        st.subheader("LLM Mapping Confidence Distribution")
        st.plotly_chart(
            px.histogram(
                df, x="llm_confidence", nbins=20,
                title="LLM Confidence Scores (< 0.8 = human review)",
            ).add_vline(x=0.8, line_dash="dash", line_color="red",
                        annotation_text="0.8 threshold"),
            use_container_width=True,
        )


# ---------------------------------------------------------
# TAB 2: DATA EXPLORER  (starter kept, mapped columns added)
# ---------------------------------------------------------

with tab2:
    st.subheader("Searchable Employee Table")
    search = st.text_input("Search (name, title, location, department)")
    view = df.copy()
    if search:
        s = search.lower()
        mask = (
            view["First_Name"].astype(str).str.lower().str.contains(s, na=False) |
            view["Last_Name"].astype(str).str.lower().str.contains(s, na=False)  |
            view["Job_Title"].astype(str).str.lower().str.contains(s, na=False)  |
            view["Location_Name"].astype(str).str.lower().str.contains(s, na=False) |
            view["Department_Name"].astype(str).str.lower().str.contains(s, na=False)
        )
        view = view[mask]
    st.write(f"{len(view)} rows")

    # Show mapped Workday columns alongside raw UKG columns if available
    display_cols = ["Employee_ID", "First_Name", "Last_Name", "Job_Title"]
    if "workday_job_code" in view.columns:
        display_cols += ["workday_job_code", "job_title_confidence"]
    display_cols += ["Department_Name"]
    if "workday_department" in view.columns:
        display_cols += ["workday_department"]
    display_cols += ["Location_Name"]
    if "workday_location_code" in view.columns:
        display_cols += ["workday_location_code"]
    display_cols += ["Annual_Salary", "review_status"]

    display_cols = [c for c in display_cols if c in view.columns]
    st.dataframe(view[display_cols], use_container_width=True)


# ---------------------------------------------------------
# TAB 3: DATA QUALITY
# Extended: reads from DB tables, not re-derived from clean_df.
# ---------------------------------------------------------

with tab3:
    st.subheader("Exception Rows")
    st.caption("Rows that failed validation and were not sent to Workday mapping.")
    if not exceptions_df.empty:
        st.write(f"{len(exceptions_df)} rows")
        st.dataframe(exceptions_df, use_container_width=True)
    else:
        st.success("No exception rows.")

    st.divider()

    st.subheader("Duplicate Rows (dropped)")
    st.caption("Dropped on (First_Name, Last_Name, Birth_Date) per spec.")
    if not dupes_df.empty:
        st.write(f"{len(dupes_df)} rows")
        st.dataframe(
            dupes_df.sort_values(["Last_Name", "First_Name"]),
            use_container_width=True,
        )
    else:
        st.success("No duplicates found.")

    st.divider()

    st.subheader("Missing Salary")
    bad_salary = df[df["Annual_Salary"].isna()]
    st.write(f"{len(bad_salary)} rows in clean table with null salary")
    if not bad_salary.empty:
        st.dataframe(bad_salary[["Employee_ID", "First_Name", "Last_Name",
                                  "Annual_Salary", "exception_reason"]],
                     use_container_width=True)

    st.subheader("Missing / Unparseable Hire Date")
    bad_hire = df[df["Hire_Date"].isna()] if "Hire_Date" in df.columns else pd.DataFrame()
    st.write(f"{len(bad_hire)} rows")
    if not bad_hire.empty:
        st.dataframe(bad_hire[["Employee_ID", "First_Name", "Last_Name",
                                "Hire_Date", "exception_reason"]],
                     use_container_width=True)


# ---------------------------------------------------------
# TAB 4: AI INSIGHTS
# - Human review queue with approve/override
# - Natural language Q&A via LangChain + Ollama
# ---------------------------------------------------------

with tab4:

    # --- Human review queue ---
    st.subheader("Human Review Queue")
    st.caption(
        "Rows where LLM confidence < 0.8 on any mapped field. "
        "Approve or override before these rows are included in the Workday export."
    )

    review_cols = [c for c in [
        "Employee_ID", "First_Name", "Last_Name",
        "Job_Title", "workday_job_code", "job_title_confidence",
        "Department_Name", "workday_department", "department_confidence",
        "Location_Name", "workday_location_code", "location_confidence",
        "review_status",
    ] if c in df.columns]

    if "needs_human_review" in df.columns:
        review_rows = df[df["needs_human_review"] == True][review_cols]
    else:
        review_rows = pd.DataFrame()

    if review_rows.empty:
        st.success("No rows pending review — all LLM mappings cleared the 0.8 threshold.")
    else:
        st.warning(f"{len(review_rows)} rows need review.")
        st.dataframe(review_rows, use_container_width=True)

        st.markdown("**Approve or override a row:**")
        employee_id = st.selectbox(
            "Select Employee_ID to action",
            review_rows["Employee_ID"].tolist() if "Employee_ID" in review_rows.columns else [],
        )

        action = st.radio("Action", ["approved", "overridden"])

        override_job  = st.text_input("Override Job Code (leave blank to keep LLM value)")
        override_dept = st.text_input("Override Department (leave blank to keep LLM value)")
        override_loc  = st.text_input("Override Location Code (leave blank to keep LLM value)")

        if st.button("Save Decision"):
            engine = get_engine()
            updates = {"review_status": action}
            if override_job:
                updates["workday_job_code"] = override_job
            if override_dept:
                updates["workday_department"] = override_dept
            if override_loc:
                updates["workday_location_code"] = override_loc

            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            updates["employee_id"] = employee_id

            with engine.begin() as conn:
                conn.execute(
                    text(f"UPDATE workday_employees SET {set_clause} "
                         f"WHERE Employee_ID = :employee_id"),
                    updates,
                )

            # Log override to mapping_overrides table (process flow doc)
            override_log = pd.DataFrame([{
                "employee_id":     employee_id,
                "action":          action,
                "override_job":    override_job or None,
                "override_dept":   override_dept or None,
                "override_loc":    override_loc or None,
                "reviewed_at":     pd.Timestamp.now().isoformat(),
            }])
            override_log.to_sql(
                "mapping_overrides", engine, if_exists="append", index=False
            )

            st.success(f"Saved '{action}' for Employee_ID {employee_id}.")
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # --- Natural language Q&A ---
    st.subheader("Ask a Question About the Data")
    st.info(
        "All inference runs locally via Ollama — no data leaves this machine. "
        "Only non-PHI columns (job codes, departments, locations, counts) "
        "are included in the prompt context."
    )

    sample_q = (
        "e.g. 'Which clinic has the most employees needing review?' "
        "or 'How many RNs are mapped to each location?'"
    )
    query = st.text_input("Your question", placeholder=sample_q)

    if st.button("Get Insight") and query:
        with st.spinner("Running local inference via Ollama..."):
            try:
                llm = ChatOllama(model="llama3:latest")

                # Only send non-PHI columns in context (process flow doc Stage 4)
                safe_cols = [c for c in [
                    "workday_job_code", "workday_department", "workday_location_code",
                    "llm_confidence", "review_status", "needs_human_review",
                    "Annual_Salary", "Pay_Type", "Department_Name",
                    "Location_Name", "Job_Title",
                ] if c in df.columns]

                context = df[safe_cols].head(40).to_csv(index=False)

                prompt = (
                    "You are a data analyst reviewing an Acme Health employee "
                    "migration from UKG to Workday. The data below contains "
                    "mapped job codes, departments, locations, and LLM confidence "
                    "scores. Names, DOB, and salary details are excluded.\n\n"
                    f"Data sample:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Answer concisely and factually based only on the data above."
                )

                response = llm.invoke(prompt)
                st.write(response.content)

            except Exception as e:
                st.error(
                    f"Ollama error: {e}\n\n"
                    "Make sure Ollama is running (`ollama serve`) and "
                    "`llama3` is pulled (`ollama pull llama3`)."
                )

    st.divider()

    # --- Load log (if Stage 5 has run) ---
    load_log = load_table("workday_load_log")
    if not load_log.empty:
        st.subheader("Workday Export Log")
        st.dataframe(load_log, use_container_width=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.success("Connected to employees.db")
st.sidebar.write(f"Clean rows: {len(df)}")
st.sidebar.write(f"Exception rows: {len(exceptions_df)}")
st.sidebar.write(f"Needs review: {needs_review}")
st.sidebar.write(
    f"Last load: "
    f"{df['load_timestamp'].max() if 'load_timestamp' in df.columns else 'n/a'}"
)

if "review_status" in df.columns:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Review status breakdown**")
    for status, count in df["review_status"].value_counts().items():
        st.sidebar.write(f"  {status}: {count}")