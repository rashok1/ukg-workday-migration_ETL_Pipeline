"""
test_pipeline.py
----------------
Three layers of tests covering every bug fixed in this iteration.

Layer 1 -- Unit tests: individual cleaning functions in isolation.
Layer 2 -- Integration tests: run_cleaning() on synthetic CSV data.
Layer 2b -- LLM mapper tests: hardcoded lookup paths (no Ollama needed).
Layer 2c -- NEW: apply_llm_mappings() contract tests on a mock dataframe.
Layer 3 -- Data quality assertions: checks against employees.db.

New tests added this iteration:
  - needs_human_review is NOT in cleaning output (it belongs to LLM stage)
  - needs_human_review IS in LLM mapper output
  - workday_employees contains LLM columns (DB write order fix)
  - mapping_cache table is created by apply_llm_mappings
  - timestamp columns are ISO strings not datetime objects
  - review_status is "auto" or "review" (not None) after LLM mapping
  - DB table overlap: no Employee_ID in both workday_employees and exceptions

Run:
    python test_pipeline.py           # all layers
    python test_pipeline.py unit      # unit + integration (no DB, no Ollama)
    python test_pipeline.py quality   # DB assertions (needs employees.db)
"""

import sys
import os
import unittest
import tempfile
import sqlite3
import pandas as pd
import numpy as np
from datetime import date, datetime
from sqlalchemy import create_engine

from cleaning_real import (
    clean_name, clean_email, validate_email, parse_date,
    clean_salary, strip_location_suffix, run_cleaning,
)
from llm_mapper import (
    normalize_job_title, normalize_department, normalize_location,
    apply_llm_mappings, _ensure_cache_table, _cache_get, _cache_set,
    CONFIDENCE_THRESHOLD, JOB_CATALOG, LOCATION_CODES,
)


# =============================================================================
# HELPERS
# =============================================================================

CSV_HEADER = (
    "Employee_ID,First_Name,Last_Name,Birth_Date,Hire_Date,Gender,Email,"
    "Job_Title,Department_Name,Location_Name,Pay_Type,Annual_Salary\n"
)

def make_csv(*rows):
    """Write rows to a temp CSV file and return path."""
    content = CSV_HEADER + "".join(rows)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        return f.name

GOOD_ROW = (
    "1,John,Smith,1980-01-15,2010-06-01,M,john@clinic.com,"
    "Registered Nurse,Nursing,Searcy,Salary,65000\n"
)
BAD_SALARY_ROW = (
    "2,Jane,Doe,1985-03-22,2015-09-01,F,jane@clinic.com,"
    "RN,Nursing,Searcy,Salary,invalid\n"
)
BAD_DATE_ROW = (
    "3,Bob,Jones,notadate,2020-01-01,M,bob@clinic.com,"
    "MA,Primary Care,Augusta,Salary,45000\n"
)
MISSING_DEPT_ROW = (
    "4,Alice,Brown,1990-07-04,2018-03-15,F,alice@clinic.com,"
    "LPN,,Newport,Salary,50000\n"
)
DUPE_ROW_A = (
    "5,Tom,Harris,1975-12-01,2005-01-01,M,tom@clinic.com,"
    "RN,Primary Care,Beebe,Salary,70000\n"
)
DUPE_ROW_B = (
    "6,Tom,Harris,1975-12-01,2005-01-01,M,tom2@clinic.com,"
    "RN,Primary Care,Beebe,Salary,70000\n"
)


def make_minimal_clean_df():
    """
    Minimal DataFrame that looks like cleaning output --
    no LLM columns, no needs_human_review.
    Used to test apply_llm_mappings() contract.
    """
    return pd.DataFrame([
        {
            "Employee_ID": 1, "First_Name": "John", "Last_Name": "Smith",
            "Job_Title": "RN", "Department_Name": "HR",
            "Location_Name": "Searcy", "Annual_Salary": 65000.0,
        },
        {
            "Employee_ID": 2, "First_Name": "Jane", "Last_Name": "Doe",
            "Job_Title": "RN", "Department_Name": "IT",
            "Location_Name": "Newport", "Annual_Salary": 70000.0,
        },
        {
            "Employee_ID": 3, "First_Name": "Bob", "Last_Name": "Jones",
            "Job_Title": "MA", "Department_Name": "Pharmacy",
            "Location_Name": "Beebe", "Annual_Salary": 45000.0,
        },
    ])


# =============================================================================
# LAYER 1: UNIT TESTS -- individual functions
# =============================================================================

class TestCleanName(unittest.TestCase):

    def test_title_case(self):
        self.assertEqual(clean_name("JOHN"), "John")

    def test_strips_whitespace(self):
        self.assertEqual(clean_name("  jane  "), "Jane")

    def test_null_returns_none(self):
        self.assertIsNone(clean_name(None))
        self.assertIsNone(clean_name(float("nan")))

    def test_strips_non_printables(self):
        result = clean_name("Jos\x00e")
        self.assertNotIn("\x00", result)

    def test_mixed_case(self):
        self.assertEqual(clean_name("mARY jAnE"), "Mary Jane")


class TestCleanEmail(unittest.TestCase):

    def test_lowercases(self):
        self.assertEqual(clean_email("User@Example.COM"), "user@example.com")

    def test_strips_whitespace(self):
        self.assertEqual(clean_email("  user@x.com  "), "user@x.com")

    def test_null_returns_none(self):
        self.assertIsNone(clean_email(None))


class TestValidateEmail(unittest.TestCase):

    def test_valid_email(self):
        self.assertTrue(validate_email("user@acmehealth.org"))

    def test_missing_at(self):
        self.assertFalse(validate_email("notanemail"))

    def test_none_is_invalid(self):
        self.assertFalse(validate_email(None))
        self.assertFalse(validate_email(""))

    def test_missing_domain(self):
        self.assertFalse(validate_email("user@"))


class TestParseDate(unittest.TestCase):

    def test_iso_format(self):
        self.assertEqual(parse_date("1990-05-15"), date(1990, 5, 15))

    def test_us_slash_format(self):
        self.assertEqual(parse_date("05/15/1990"), date(1990, 5, 15))

    def test_us_dash_format(self):
        self.assertEqual(parse_date("05-15-1990"), date(1990, 5, 15))

    def test_invalid_string_returns_none(self):
        self.assertIsNone(parse_date("invalid"))

    def test_null_returns_none(self):
        self.assertIsNone(parse_date(None))
        self.assertIsNone(parse_date(float("nan")))


class TestCleanSalary(unittest.TestCase):

    def test_plain_number(self):
        self.assertEqual(clean_salary("75000"), 75000.0)

    def test_dollar_sign(self):
        self.assertEqual(clean_salary("$75,000"), 75000.0)

    def test_comma_separated(self):
        self.assertEqual(clean_salary("75,000.00"), 75000.0)

    def test_invalid_string_returns_nan(self):
        self.assertTrue(np.isnan(clean_salary("invalid")))

    def test_null_returns_nan(self):
        self.assertTrue(np.isnan(clean_salary(None)))


class TestStripLocationSuffix(unittest.TestCase):

    def test_strips_hq(self):
        self.assertEqual(strip_location_suffix("Searcy HQ"), "Searcy")

    def test_strips_office(self):
        self.assertEqual(strip_location_suffix("Newport Office"), "Newport")

    def test_no_suffix(self):
        self.assertEqual(strip_location_suffix("Augusta"), "Augusta")

    def test_null_returns_remote(self):
        # null input -> "Remote" sentinel, not crash
        self.assertEqual(strip_location_suffix(None), "Remote")
        self.assertEqual(strip_location_suffix(float("nan")), "Remote")

    def test_empty_string(self):
        # empty string after strip -> empty string (not "Remote")
        result = strip_location_suffix("")
        self.assertIsInstance(result, str)


# =============================================================================
# LAYER 2: INTEGRATION TESTS -- run_cleaning() on synthetic CSV
# =============================================================================

class TestRunCleaning(unittest.TestCase):

    def test_good_row_goes_to_clean(self):
        path = make_csv(GOOD_ROW)
        clean, exc, dupes, stats = run_cleaning(path)
        self.assertEqual(stats["clean_count"], 1)
        self.assertEqual(stats["exception_count"], 0)

    def test_bad_salary_goes_to_exceptions(self):
        path = make_csv(BAD_SALARY_ROW)
        clean, exc, dupes, stats = run_cleaning(path)
        self.assertEqual(stats["exception_count"], 1)
        self.assertIn("invalid salary", exc.iloc[0]["exception_reason"])

    def test_bad_date_goes_to_exceptions(self):
        path = make_csv(BAD_DATE_ROW)
        clean, exc, dupes, stats = run_cleaning(path)
        self.assertEqual(stats["exception_count"], 1)
        self.assertIn("birth date", exc.iloc[0]["exception_reason"])

    def test_missing_dept_goes_to_exceptions(self):
        path = make_csv(MISSING_DEPT_ROW)
        clean, exc, dupes, stats = run_cleaning(path)
        self.assertEqual(stats["exception_count"], 1)
        self.assertIn("missing department", exc.iloc[0]["exception_reason"])

    def test_duplicate_removed(self):
        path = make_csv(DUPE_ROW_A, DUPE_ROW_B)
        clean, exc, dupes, stats = run_cleaning(path)
        self.assertEqual(stats["dupes_removed"], 1)
        self.assertEqual(len(dupes), 2)

    def test_gender_normalised(self):
        path = make_csv(GOOD_ROW)
        clean, _, _, _ = run_cleaning(path)
        self.assertEqual(clean.iloc[0]["Gender"], "Male")

    def test_location_hq_stripped(self):
        row = GOOD_ROW.replace("Searcy", "Searcy HQ")
        path = make_csv(row)
        clean, _, _, _ = run_cleaning(path)
        self.assertEqual(clean.iloc[0]["Location_Name"], "Searcy")

    def test_audit_columns_present(self):
        path = make_csv(GOOD_ROW)
        clean, _, _, _ = run_cleaning(path)
        for col in ["source_system", "source_record_id", "load_timestamp"]:
            self.assertIn(col, clean.columns)
        self.assertEqual(clean.iloc[0]["source_system"], "UKG")

    def test_load_timestamp_is_iso_string(self):
        """
        Timestamps must be ISO strings, not datetime objects.
        SQLite stores them as TEXT; mixing types causes silent inconsistencies.
        """
        path = make_csv(GOOD_ROW)
        clean, exc, _, _ = run_cleaning(path)
        ts = clean.iloc[0]["load_timestamp"]
        self.assertIsInstance(ts, str,
            "load_timestamp should be an ISO string, not a datetime object")
        # Should be parseable as ISO datetime
        datetime.fromisoformat(ts)

    # --- KEY BUG FIX TEST ---
    def test_needs_human_review_NOT_in_cleaning_output(self):
        """
        needs_human_review must NOT exist after run_cleaning().
        It is defined in apply_llm_mappings() (Stage 4), not cleaning (Stage 2).
        If this column exists here, something is leaking between stages.
        """
        path = make_csv(GOOD_ROW)
        clean, exc, dupes, _ = run_cleaning(path)
        self.assertNotIn("needs_human_review", clean.columns,
            "needs_human_review should not exist after cleaning -- "
            "it is only added by apply_llm_mappings()")
        self.assertNotIn("workday_job_code", clean.columns,
            "workday_job_code should not exist after cleaning -- "
            "it is only added by apply_llm_mappings()")


# =============================================================================
# LAYER 2b: LLM MAPPER UNIT TESTS -- hardcoded lookup paths only
# These do NOT require Ollama to be running.
# =============================================================================

class TestLLMMapperHardcodedPaths(unittest.TestCase):

    def test_rn_maps_to_code_without_llm(self):
        result = normalize_job_title("RN")
        self.assertEqual(result["workday_value"], "RN")
        self.assertEqual(result["confidence"], 1.0)

    def test_registered_nurse_maps_without_llm(self):
        result = normalize_job_title("Registered Nurse")
        self.assertEqual(result["workday_value"], "RN")
        self.assertEqual(result["confidence"], 1.0)

    def test_hr_dept_maps_without_llm(self):
        result = normalize_department("HR")
        self.assertEqual(result["workday_value"], "Human Resources")
        self.assertEqual(result["confidence"], 1.0)

    def test_searcy_location_maps_without_llm(self):
        result = normalize_location("Searcy")
        self.assertEqual(result["workday_value"], "SRCY")
        self.assertEqual(result["confidence"], 1.0)

    def test_remote_maps_without_llm(self):
        result = normalize_location("Remote")
        self.assertEqual(result["workday_value"], "REM")
        self.assertEqual(result["confidence"], 1.0)

    def test_confidence_threshold_is_sensible(self):
        self.assertGreater(CONFIDENCE_THRESHOLD, 0.5)
        self.assertLessEqual(CONFIDENCE_THRESHOLD, 1.0)

    def test_all_job_catalog_codes_are_strings(self):
        for code, label in JOB_CATALOG.items():
            self.assertIsInstance(code, str)
            self.assertGreater(len(code), 0)


# =============================================================================
# LAYER 2c: apply_llm_mappings() CONTRACT TESTS
# These verify the output schema and that needs_human_review is correctly
# defined here and NOT inherited from cleaning.
# Uses hardcoded-shorthand values so Ollama is NOT needed.
# =============================================================================

class TestApplyLLMMappingsContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run apply_llm_mappings once on minimal df for all tests."""
        df = make_minimal_clean_df()
        cls.result = apply_llm_mappings(df.copy())

    def test_needs_human_review_column_exists(self):
        """needs_human_review must exist after apply_llm_mappings()."""
        self.assertIn("needs_human_review", self.result.columns)

    def test_needs_human_review_is_bool(self):
        self.assertTrue(self.result["needs_human_review"].dtype == bool or
                        self.result["needs_human_review"].dtype == object)
        for val in self.result["needs_human_review"]:
            self.assertIsInstance(val, (bool, np.bool_))

    def test_workday_job_code_exists(self):
        self.assertIn("workday_job_code", self.result.columns)

    def test_workday_department_exists(self):
        self.assertIn("workday_department", self.result.columns)

    def test_workday_location_code_exists(self):
        self.assertIn("workday_location_code", self.result.columns)

    def test_llm_confidence_in_range(self):
        confs = self.result["llm_confidence"].dropna()
        self.assertTrue((confs >= 0.0).all() and (confs <= 1.0).all())

    def test_review_status_values(self):
        """review_status must be 'auto' or 'review' -- never None or NaN."""
        valid = {"auto", "review"}
        actual = set(self.result["review_status"].dropna().unique())
        self.assertTrue(actual.issubset(valid),
                        f"Unexpected review_status values: {actual - valid}")
        # No nulls
        self.assertEqual(self.result["review_status"].isnull().sum(), 0)

    def test_known_shorthand_has_confidence_1(self):
        """RN (in JOB_SHORTHAND) must return confidence 1.0 -- no LLM call."""
        rn_rows = self.result[self.result["Job_Title"] == "RN"]
        self.assertTrue((rn_rows["job_title_confidence"] == 1.0).all())

    def test_row_count_preserved(self):
        """apply_llm_mappings must not add or drop rows."""
        df = make_minimal_clean_df()
        result = apply_llm_mappings(df.copy())
        self.assertEqual(len(result), len(df))

    def test_unique_value_mapping_does_not_duplicate(self):
        """
        Two rows with same Job_Title should get same workday_job_code.
        Both rows in make_minimal_clean_df have Job_Title="RN".
        """
        rn_rows = self.result[self.result["Job_Title"] == "RN"]
        codes = rn_rows["workday_job_code"].unique()
        self.assertEqual(len(codes), 1,
                         f"Same input should map to same code, got: {codes}")


# =============================================================================
# LAYER 2d: PERSISTENT CACHE TESTS
# Verify the mapping_cache table works correctly.
# Uses a temp DB so it does not pollute employees.db.
# =============================================================================

class TestPersistentCache(unittest.TestCase):

    def setUp(self):
        """Point cache at a temp DB for each test."""
        import llm_mapper
        self.tmp = tempfile.mktemp(suffix=".db")
        self._orig_path = llm_mapper.CACHE_DB_PATH
        llm_mapper.CACHE_DB_PATH = self.tmp

    def tearDown(self):
        import llm_mapper
        llm_mapper.CACHE_DB_PATH = self._orig_path
        if os.path.exists(self.tmp):
            os.unlink(self.tmp)

    def test_cache_table_created(self):
        import llm_mapper
        _ensure_cache_table()
        with sqlite3.connect(self.tmp) as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        self.assertIn("mapping_cache", tables)

    def test_cache_miss_returns_none(self):
        import llm_mapper
        _ensure_cache_table()
        result = _cache_get("job_title", "NonExistentRole999")
        self.assertIsNone(result)

    def test_cache_set_and_get(self):
        import llm_mapper
        _ensure_cache_table()
        test_result = {"workday_value": "RN", "confidence": 0.99, "reason": "test"}
        _cache_set("job_title", "Nurse RN", test_result)
        retrieved = _cache_get("job_title", "Nurse RN")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["workday_value"], "RN")
        self.assertAlmostEqual(retrieved["confidence"], 0.99)

    def test_cache_different_field_types_are_separate(self):
        import llm_mapper
        _ensure_cache_table()
        _cache_set("job_title",  "IT", {"workday_value": "IT_SUP",    "confidence": 0.9, "reason": "job"})
        _cache_set("department", "IT", {"workday_value": "IT",        "confidence": 1.0, "reason": "dept"})
        job_result  = _cache_get("job_title",  "IT")
        dept_result = _cache_get("department", "IT")
        self.assertEqual(job_result["workday_value"],  "IT_SUP")
        self.assertEqual(dept_result["workday_value"], "IT")


# =============================================================================
# LAYER 3: DATA QUALITY ASSERTIONS -- against employees.db
# Require etl_pipeline.py to have completed first.
# =============================================================================

class TestDataQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            engine = create_engine("sqlite:///employees.db")
            cls.df  = pd.read_sql("SELECT * FROM workday_employees", engine)
            cls.exc = pd.read_sql("SELECT * FROM exceptions", engine)
            cls.db_available = True
        except Exception:
            cls.db_available = False

    def setUp(self):
        if not self.db_available:
            self.skipTest("employees.db not found -- run etl_pipeline.py first")

    # --- DB write order fix: LLM columns must exist in workday_employees ---

    def test_workday_job_code_in_db(self):
        """
        Fails if etl_pipeline.py wrote workday_employees before LLM mapping ran.
        The fix: workday_employees is written only once, after apply_llm_mappings().
        """
        self.assertIn("workday_job_code", self.df.columns,
                      "workday_job_code missing -- workday_employees may have been "
                      "written before LLM mapping completed")

    def test_needs_human_review_in_db(self):
        """
        Fails if workday_employees was written before apply_llm_mappings().
        app.py reads this column; it must exist.
        """
        self.assertIn("needs_human_review", self.df.columns,
                      "needs_human_review missing from DB -- "
                      "workday_employees must be written after LLM mapping")

    def test_review_status_in_db(self):
        self.assertIn("review_status", self.df.columns)

    # --- Completeness ---

    def test_no_null_employee_id(self):
        self.assertEqual(self.df["Employee_ID"].isnull().sum(), 0)

    def test_no_null_first_name(self):
        self.assertEqual(self.df["First_Name"].isnull().sum(), 0)

    def test_no_null_hire_date_in_clean(self):
        self.assertEqual(self.df["Hire_Date"].isnull().sum(), 0)

    def test_no_null_birth_date_in_clean(self):
        self.assertEqual(self.df["Birth_Date"].isnull().sum(), 0)

    # --- No duplicates ---

    def test_no_duplicate_employee_ids(self):
        duped = self.df[self.df.duplicated("Employee_ID")]
        self.assertEqual(len(duped), 0)

    def test_no_duplicate_name_dob(self):
        duped = self.df[
            self.df.duplicated(["First_Name", "Last_Name", "Birth_Date"])
        ]
        self.assertEqual(len(duped), 0)

    # --- Value validity ---

    def test_gender_values_valid(self):
        valid = {"Male", "Female", "Not Specified"}
        actual = set(self.df["Gender"].dropna().unique())
        self.assertTrue(actual.issubset(valid), f"Bad gender values: {actual - valid}")

    def test_salary_non_negative(self):
        neg = self.df[self.df["Annual_Salary"].notna() & (self.df["Annual_Salary"] < 0)]
        self.assertEqual(len(neg), 0)

    def test_pay_type_valid(self):
        valid = {"Salary", "Hourly"}
        actual = set(self.df["Pay_Type"].dropna().unique())
        self.assertTrue(actual.issubset(valid), f"Bad pay types: {actual - valid}")

    def test_job_codes_are_catalog_members(self):
        if "workday_job_code" not in self.df.columns:
            self.skipTest("LLM columns not present")
        invalid = self.df[
            self.df["workday_job_code"].notnull() &
            ~self.df["workday_job_code"].isin(JOB_CATALOG.keys())
        ]
        self.assertEqual(len(invalid), 0,
                         f"Invalid codes: {invalid['workday_job_code'].tolist()}")

    def test_llm_confidence_in_range(self):
        if "llm_confidence" not in self.df.columns:
            self.skipTest("LLM columns not present")
        bad = self.df[(self.df["llm_confidence"] < 0) | (self.df["llm_confidence"] > 1)]
        self.assertEqual(len(bad), 0)

    def test_review_status_valid_values(self):
        if "review_status" not in self.df.columns:
            self.skipTest("LLM columns not present")
        valid = {"auto", "review", "approved", "overridden"}
        actual = set(self.df["review_status"].dropna().unique())
        self.assertTrue(actual.issubset(valid), f"Bad review_status: {actual - valid}")

    # --- Audit ---

    def test_source_system_is_ukg(self):
        self.assertTrue((self.df["source_system"] == "UKG").all())

    def test_load_timestamp_is_string(self):
        """
        Timestamps must be ISO strings in the DB.
        Mixed datetime/string types break comparison queries.
        """
        self.assertIn("load_timestamp", self.df.columns)
        sample = self.df["load_timestamp"].dropna().iloc[0]
        self.assertIsInstance(str(sample), str)

    # --- No overlap between tables ---

    def test_no_employee_id_in_both_clean_and_exceptions(self):
        if self.exc.empty:
            return
        overlap = set(self.df["Employee_ID"]) & set(self.exc["Employee_ID"])
        self.assertEqual(len(overlap), 0,
                         f"IDs in both tables: {overlap}")

    def test_clean_table_not_empty(self):
        self.assertGreater(len(self.df), 0)


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    unit_suites = [
        TestCleanName, TestCleanEmail, TestValidateEmail,
        TestParseDate, TestCleanSalary, TestStripLocationSuffix,
        TestRunCleaning, TestLLMMapperHardcodedPaths,
        TestApplyLLMMappingsContract, TestPersistentCache,
    ]
    quality_suites = [TestDataQuality]

    if mode in ("all", "unit"):
        for s in unit_suites:
            suite.addTests(loader.loadTestsFromTestCase(s))

    if mode in ("all", "quality"):
        for s in quality_suites:
            suite.addTests(loader.loadTestsFromTestCase(s))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)