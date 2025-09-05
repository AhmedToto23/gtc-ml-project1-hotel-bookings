# =============================
# 📌 Hotel Bookings Data Project
# =============================

# --- Imports ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# --- Load Dataset ---
df = pd.read_csv("hotel_bookings.csv")
print("Dataset shape: ", df.shape)

# --- Basic Info ---
df.info()
print(df.describe())
print(df.describe(include='all'))

# --- Missing values check ---
print(df.isnull().sum())
missing = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing / len(df) * 100).round(2)
print(missing_percentage)

# --- Duplicate values check ---
for i in df.columns:
    print(i, "=>", df[i].duplicated().sum())

# =============================
# 📌 Data Quality Report
# =============================
def build_column_stats(series: pd.Series) -> dict:
    total = len(series)
    dt = str(series.dtype)
    non_null_cnt = series.count()
    missing_cnt = series.isnull().sum()
    missing_pct = (missing_cnt / total * 100).round(2)
    uniq_cnt = series.nunique(dropna=True)
    top = None if series.dropna().empty else series.dropna().value_counts().idxmax()
    return {
        "dtype": dt,
        "non_null": int(non_null_cnt),
        "missing": int(missing_cnt),
        "missing_pct": float(missing_pct),
        "unique": int(uniq_cnt),
        "top_value": top,
    }

def data_quality_report_alt(df: pd.DataFrame) -> pd.DataFrame:
    stats_list = []
    for name, ser in df.items():
        col_stats = build_column_stats(ser)
        stats_list.append({
            "column": name,
            "dtype": col_stats["dtype"],
            "non_null": col_stats["non_null"],
            "missing": col_stats["missing"],
            "missing_pct": col_stats["missing_pct"],
            "unique": col_stats["unique"],
            "top_value": col_stats["top_value"],
        })
    report_df = pd.DataFrame(stats_list)
    return report_df.sort_values(by="column").reset_index(drop=True)

report = data_quality_report_alt(df)
report.to_csv("data_quality_report.csv", index=False)
print("✅ Saved data_quality_report.csv")

# --- Quick Issues Printer ---
def collect_issues_from_quality(quality_df: pd.DataFrame, orig_df: pd.DataFrame) -> List[str]:
    issues_list: List[str] = []
    total_rows = len(orig_df)

    # Missing values > 30%
    miss_thresh = 30.0
    highly_missing = quality_df.loc[quality_df["missing_pct"] > miss_thresh, "column"].tolist()
    if highly_missing:
        issues_list.append(f"Columns with >{miss_thresh}% missing: {highly_missing}")

    # High cardinality (>50% unique)
    card_thresh = total_rows * 0.5
    high_card_cols = quality_df.loc[quality_df["unique"] > card_thresh, "column"].tolist()
    if high_card_cols:
        issues_list.append(f"Columns with very high cardinality (>50% rows unique): {high_card_cols}")

    # Print results
    if issues_list:
        print("⚠️ Quick issues detected:")
        for item in issues_list:
            print("-", item)
    else:
        print("✅ No obvious automatic flags.")
    return issues_list

issues = collect_issues_from_quality(report, df)

# =============================
# 📌 Visualization
# =============================

# Distribution plots
palette = sns.color_palette("husl", len(df.select_dtypes(include="number").columns))
for idx, col in enumerate(df.select_dtypes(include="number").columns):
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col, color=palette[idx], edgecolor="black", kde=True)
    plt.title(f"Distribution of {col}", fontsize=14, fontweight="bold")
    plt.grid()
    plt.show()

# Missing values heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing values heatmap")
plt.show()

# Outliers check
for col in df.select_dtypes(include="number"):
    plt.figure(figsize=(10, 3))
    sns.boxplot(data=df, x=df[col], palette="Set2")
    plt.title(f"Boxplot of {col}", fontsize=14)
    plt.show()

    plt.figure(figsize=(10, 3))
    sns.histplot(df[col].dropna(), bins=50)
    plt.title(f"Histogram of {col}")
    plt.show()

# =============================
# 📌 Data Cleaning
# =============================
df_copy = df.copy()

# Impute numeric columns with mean
for col in df_copy.select_dtypes(include="number").columns:
    df_copy[col].fillna(df_copy[col].mean(), inplace=True)

# Impute country with mode
df_copy["country"].fillna(df_copy["country"].mode()[0], inplace=True)

# Drop duplicates
df_copy = df_copy.drop_duplicates()
print("Duplicates after removal:", df_copy.duplicated().sum())

# Correct dtypes
print("Correcting data type...")
df_copy["reservation_status_date"] = pd.to_datetime(df_copy["reservation_status_date"], errors="coerce")
df_copy["children"] = df_copy["children"].astype("Int64")
print(df_copy[["reservation_status_date", "children"]].dtypes)

# --- Outlier handling ---
outliers_col = []
for col in df_copy.select_dtypes(include="number").columns:
    q1 = df_copy[col].quantile(0.25)
    q3 = df_copy[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    count_outliers = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)].shape[0]
    if count_outliers > 0:
        outliers_col.append(col)

def outliers_handling(df_copy: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        s = df_copy[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
        df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound

outliers_handling(df_copy, cols=outliers_col)

# =============================
# 📌 Feature Engineering
# =============================
df_fet = df_copy.copy()

# Total guests
df_fet["total_guests"] = df_fet["adults"] + df_fet["children"].fillna(0) + df_fet["babies"]

# Total nights
df_fet["total_nights"] = df_fet["stays_in_weekend_nights"] + df_fet["stays_in_week_nights"]

# Family flag
df_fet["is_family"] = np.where((df_fet["children"] + df_fet["babies"]) > 0, 1, 0)

# =============================
# 📌 Encoding
# =============================
df_copy = pd.get_dummies(df_copy, columns=["meal", "market_segment"], drop_first=True)

# Country encoding
country_freq = df["country"].value_counts(normalize=True)
df_copy["country_encoded"] = df_copy["country"].map(country_freq)
rare_countries = country_freq[country_freq < 0.01].index
df_copy["country_grouped"] = df_copy["country"].replace(rare_countries, "Other")

# Drop unnecessary columns
df_copy = df_copy.drop(["reservation_status", "reservation_status_date"], axis=1)

# =============================
# 📌 Train-Test Split
# =============================
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ All datasets saved!")
