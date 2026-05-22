"""
test_pipeline.py
----------------
README requirement: "how you'd test it (unit tests, data quality assertions)"

Three layers of tests:
  1. Unit tests  -- individual cleaning functions in isolation
  2. Integration tests -- full run_cleaning() on synthetic data
  3. Data quality assertions -- checks on the actual SQLite output

Run:
    python test_pipeline.py          # all tests
    python test_pipeline.py unit     # unit only
    python test_pipeline.py quality  # DB quality checks only (needs employees.db)
"""

import sys
import json
import unittest
import pandas as pd
import numpy as np
from io import StringIO
from datetime import date
from sqlalchemy import create_engine

from cleaning_real import (
    clean_name, clean_email, validate_email, parse_date, clean_salary, run_cleaning
)
from llm_mapper import (
    normalize_job_title, normalize_department, normalize_location,
    CONFIDENCE_THRESHOLD, JOB_CATALOG, VALID_LOCATION_CODES,
)


# =============================================================================
# LAYER 1: UNIT TESTS
# Test each cleaning function in complete isolation -- no DB, no LLM, no CSV.
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
        # Non-ASCII bytes should be stripped
        result = clean_name("Jos\x00e")
        self.assertNotIn("\x00", result)


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


# =============================================================================
# LAYER 2: INTEGRATION TESTS
# Run run_cleaning() against synthetic CSV data and assert on the output.
# No real file needed -- we construct the CSV in-memory.
# =============================================================================

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

CSV_HEADER = (
    "Employee_ID,First_Name,Last_Name,Birth_Date,Hire_Date,Gender,Email,"
    "Job_Title,Department_Name,Location_Name,Pay_Type,Annual_Salary\n"
)


def make_csv(*rows):
    """Build an in-memory CSV path substitute using StringIO."""
    content = CSV_HEADER + "".join(rows)
    # write to a temp file since run_cleaning takes a path
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False) as f:
        f.write(content)
        return f.name


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
        self.assertEqual(len(dupes), 2)  # both copies logged to dupes table

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


# =============================================================================
# LAYER 2b: LLM MAPPER UNIT TESTS (hardcoded lookup paths only -- no Ollama)
# These test the lookup layer without needing Ollama running.
# =============================================================================

class TestLLMMapperHardcodedPaths(unittest.TestCase):
    """
    Tests the hardcoded lookup path only.
    LLM path tests would require Ollama running -- keep those manual / separate.
    """

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

    def test_all_job_catalog_codes_are_valid(self):
        # Every key in JOB_CATALOG should be a non-empty string
        for code, label in JOB_CATALOG.items():
            self.assertIsInstance(code, str)
            self.assertGreater(len(code), 0)
            self.assertIsInstance(label, str)

    def test_confidence_threshold_is_sensible(self):
        self.assertGreater(CONFIDENCE_THRESHOLD, 0.5)
        self.assertLessEqual(CONFIDENCE_THRESHOLD, 1.0)


# =============================================================================
# LAYER 3: DATA QUALITY ASSERTIONS
# Run against the actual employees.db after etl_pipeline.py has run.
# These are the "data quality assertions" the README asks for.
# Mirrors what a Great Expectations or dbt test suite would check.
# =============================================================================

class TestDataQuality(unittest.TestCase):
    """
    Requires employees.db to exist (run etl_pipeline.py first).
    Skips gracefully if the DB isn't present.
    """

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

    # --- Completeness ---

    def test_no_null_employee_id(self):
        self.assertEqual(self.df["Employee_ID"].isnull().sum(), 0)

    def test_no_null_first_name(self):
        self.assertEqual(self.df["First_Name"].isnull().sum(), 0)

    def test_no_null_last_name(self):
        self.assertEqual(self.df["Last_Name"].isnull().sum(), 0)

    def test_no_null_hire_date_in_clean(self):
        # clean table should have no null hire dates (those go to exceptions)
        self.assertEqual(self.df["Hire_Date"].isnull().sum(), 0)

    def test_no_null_birth_date_in_clean(self):
        self.assertEqual(self.df["Birth_Date"].isnull().sum(), 0)

    # --- No duplicates in clean table ---

    def test_no_duplicate_employee_ids(self):
        duped = self.df[self.df.duplicated("Employee_ID")]
        self.assertEqual(len(duped), 0,
                         f"Duplicate Employee_IDs: {duped['Employee_ID'].tolist()}")

    def test_no_duplicate_name_dob(self):
        duped = self.df[
            self.df.duplicated(["First_Name", "Last_Name", "Birth_Date"])
        ]
        self.assertEqual(len(duped), 0)

    # --- Value validity ---

    def test_gender_values_valid(self):
        valid = {"Male", "Female", "Not Specified"}
        actual = set(self.df["Gender"].dropna().unique())
        self.assertTrue(actual.issubset(valid),
                        f"Unexpected gender values: {actual - valid}")

    def test_salary_non_negative_where_present(self):
        neg = self.df[self.df["Annual_Salary"].notna() &
                      (self.df["Annual_Salary"] < 0)]
        self.assertEqual(len(neg), 0)

    def test_pay_type_valid_values(self):
        valid = {"Salary", "Hourly"}
        actual = set(self.df["Pay_Type"].dropna().unique())
        self.assertTrue(actual.issubset(valid),
                        f"Unexpected Pay_Type values: {actual - valid}")

    # --- LLM mapping columns present and sane ---

    def test_workday_job_code_column_exists(self):
        self.assertIn("workday_job_code", self.df.columns)

    def test_job_codes_are_valid_catalog_values(self):
        if "workday_job_code" not in self.df.columns:
            self.skipTest("LLM mapping not yet run")
        invalid = self.df[
            ~self.df["workday_job_code"].isin(JOB_CATALOG.keys()) &
            self.df["workday_job_code"].notnull()
        ]
        self.assertEqual(len(invalid), 0,
                         f"Invalid job codes: {invalid['workday_job_code'].tolist()}")

    def test_llm_confidence_in_range(self):
        if "llm_confidence" not in self.df.columns:
            self.skipTest("LLM mapping not yet run")
        out_of_range = self.df[
            (self.df["llm_confidence"] < 0) |
            (self.df["llm_confidence"] > 1)
        ]
        self.assertEqual(len(out_of_range), 0)

    def test_review_status_values_valid(self):
        if "review_status" not in self.df.columns:
            self.skipTest("LLM mapping not yet run")
        valid = {"auto", "review", "approved", "overridden"}
        actual = set(self.df["review_status"].dropna().unique())
        self.assertTrue(actual.issubset(valid),
                        f"Unexpected review_status values: {actual - valid}")

    # --- Audit columns ---

    def test_source_system_is_ukg(self):
        self.assertTrue((self.df["source_system"] == "UKG").all())

    def test_source_record_id_matches_employee_id(self):
        mismatches = self.df[
            self.df["source_record_id"] != self.df["Employee_ID"]
        ]
        self.assertEqual(len(mismatches), 0)

    def test_load_timestamp_present(self):
        self.assertIn("load_timestamp", self.df.columns)
        self.assertEqual(self.df["load_timestamp"].isnull().sum(), 0)

    # --- Referential integrity: exceptions don't overlap clean ---

    def test_no_overlap_clean_exceptions(self):
        if self.exc.empty:
            return
        overlap = set(self.df["Employee_ID"]) & set(self.exc["Employee_ID"])
        self.assertEqual(len(overlap), 0,
                         f"Employee_IDs in both tables: {overlap}")

    # --- Summary assertion ---

    def test_total_row_conservation(self):
        """
        Clean + exceptions + dupes_removed should account for all raw rows.
        We don't have the raw count here but we can assert clean > 0.
        """
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
        TestParseDate, TestCleanSalary, TestRunCleaning,
        TestLLMMapperHardcodedPaths,
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