import pandas as pd
df = pd.read_csv("ukg_employees_raw.csv")

#Trim, title-case, strip non-printables from the doc
#Fixing Ttile Case and Stripping Non-Printables

def clean_name(name):
    if pd.isnull(name):
        return None

    return (
        str(name)
        .strip()
        .title()
    )

df["First_Name"] = df["First_Name"].apply(clean_name)
df["Last_Name"] = df["Last_Name"].apply(clean_name)

#Parse mixed formats to all YYYY-MM-DD
from dateutil import parser

parsed_dates = pd.to_datetime(
    df["Birth_Date"],
    errors="coerce"
)

bad_dates = df[
    parsed_dates.isnull()
]

print(bad_dates)
def parse_date(value):
    try:
        return parser.parse(str(value)).date()
    except:
        return None
    

df["Birth_Date"] = df["Birth_Date"].apply(parse_date)
df["Hire_Date"] = df["Hire_Date"].apply(parse_date)

#M/F -> Male/Female, blank -> Not Specified

gender_map = {
    "M": "Male",
    "F": "Female"
}

df["Gender"] = (
    df["Gender"]
    .map(gender_map)
    .fillna("Not Specified")
)

#Lowercase; validate; flag missing
#if the email is missing, we can create a flag for it, and we can also lowercase the email for consistency. but what if their email contains uppercase letters? 
df["Email"] = (
    df["Email"]
    .astype(str)
    .str.strip()
    .str.lower()
)

#validating the emails
import re

EMAIL_REGEX = r'^[\w\.-]+@[\w\.-]+\.\w+$'

df["email_valid"] = df["Email"].apply(
    lambda x: bool(re.match(EMAIL_REGEX, x))
)


#converting salary to numeric, and flagging invalid entries
def clean_salary(value):
    if pd.isnull(value):
        return None

    value = str(value).lower()

    value = value.replace("$", "").replace(",", "")

    if "k" in value:
        value = value.replace("k", "")
        return float(value) * 1000

    try:
        return float(value)
    except:
        return None
    
df["Annual_Salary"] = df["Annual_Salary"].apply(clean_salary)

# Fill with sentinel + add to exception report for missing values --> unknown
######## come back herer

# removing dupliactes on first name, last name, and birth date.
df["Department_Name"] = (
    df["Department_Name"]
    .fillna("Unknown")
)

# exceptions table to seperate vlaid roels from problematic ones. 


