import pandas as pd
df = pd.read_csv("ukg_employees_raw.csv")

print(df.columns)
print(df.info())
print(df.head())

print(df["Location_Name"].unique())
print(df["Location_Name"].value_counts())

df.isnull().sum()

#unique values in categorical columns
for col in [
    "Gender",
    "Pay_Type",
    "Department_Name",
    "Location_Name",
    "Job_Title"
]:
    print("\n", col)
    print(df[col].value_counts(dropna=False))

print(df["Annual_Salary"].unique())

#dupliactes
duplicates = df[
    df.duplicated(
        subset=[
            "First_Name",
            "Last_Name",
            "Birth_Date"
        ],
        keep=False
    )
]

print(duplicates)

