"""
Preprocessing utilities for the Used Car Price Prediction project.

This module contains the shared preprocessing logic used by both:
1. Regression: predicting the continuous target `price`
2. Classification: predicting the categorical target `price_class`

Main leakage-prevention rule:
- Text parsing that uses only each row's own values can be done before splitting.
- Statistical rules such as medians, IQR thresholds, and price_class cutoffs
  must be calculated from the train set only.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_ORDER = ["low", "mid", "high"]

NUM_COLS = ["model_year", "milage", "HP", "engine_L", "cylinders"]
CAT_COLS = ["fuel_type", "accident", "clean_title", "turbo", "transmission_type"]

DROP_COLS = ["brand", "model", "engine", "transmission", "ext_col", "int_col"]


# ==========================================================================================
# 1. Preprocessing Before Data Splitting
# ==========================================================================================
def preprocess_before_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning before train/test splitting.

    This step uses only row-level text information, so it does not use train/test
    distribution statistics.

    Processing steps:
    - Strip leading/trailing spaces from column names and object-type values.
    - Remove invalid placeholder rows where fuel_type or engine is recorded as "–".
    - Normalize fuel_type, accident, and clean_title missing values.
    - Convert price and milage from object/string format to numeric format.
    - Extract HP, engine_L, cylinders, and turbo from the engine text.
    - Simplify transmission text into transmission_type.
    - Drop rows without essential values: price, milage, model_year.
    """
    df = df.copy()

    # 1-1. Remove unnecessary spaces from column names and string values.
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 1-2. Remove rows with invalid placeholder values.
    df = df[df["fuel_type"] != "–"]
    df = df[df["engine"] != "–"]

    # 1-3. Handle structural missing values in categorical columns.
    # fuel_type: "not supported" is treated as missing, then filled as Electric.
    # accident: missing does not necessarily mean "no accident", so use Unknown.
    # clean_title: only Yes exists as explicit value, so missing is treated as No.
    df["fuel_type"] = df["fuel_type"].replace("not supported", np.nan)
    df["fuel_type"] = df["fuel_type"].fillna("Electric")
    df["accident"] = df["accident"].fillna("Unknown")
    df["clean_title"] = df["clean_title"].fillna("No")

    # 1-4. Convert price and milage into numeric values.
    # Examples:
    # "$10,300" -> 10300
    # "51,000 mi." -> 51000
    df["price"] = df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["milage"] = df["milage"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
    df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

    # 1-5. Extract engine-related features.
    engine = df["engine"].astype(str)

    # Extract horsepower, e.g., "280HP" -> 280.
    df["HP"] = pd.to_numeric(
        engine.str.extract(r"(\d+(?:\.\d+)?)\s*HP", flags=re.IGNORECASE)[0],
        errors="coerce",
    )

    # Extract engine displacement, e.g., "3.5L" -> 3.5.
    df["engine_L"] = pd.to_numeric(
        engine.str.extract(r"(\d+(?:\.\d+)?)\s*(?:L|Liter)", flags=re.IGNORECASE)[0],
        errors="coerce",
    )

    # Extract cylinder counts from multiple text patterns:
    # V6 / I4 / H6, "6 Cylinder", "Straight 6", "Flat 4".
    cyl1 = engine.str.extract(r"\b[VIH]\s*-?\s*(\d{1,2})\b", flags=re.IGNORECASE)[0]
    cyl2 = engine.str.extract(r"\b(\d{1,2})\s*Cylinder\b", flags=re.IGNORECASE)[0]
    cyl3 = engine.str.extract(r"\b(?:Straight|Flat)\s*(\d{1,2})\b", flags=re.IGNORECASE)[0]
    df["cylinders"] = pd.to_numeric(cyl1.fillna(cyl2).fillna(cyl3), errors="coerce")

    # Create a boolean turbo flag.
    df["turbo"] = engine.str.contains("turbo", case=False, na=False)

    # 1-6. Simplify transmission text.
    def get_transmission_type(x: object) -> str | float:
        if pd.isna(x):
            return np.nan

        value = str(x).lower()

        if "cvt" in value:
            return "cvt"
        if "dual" in value or "auto-shift" in value:
            return "dual_shift"
        if "manual" in value or "m/t" in value:
            return "manual"
        if "automatic" in value or "a/t" in value:
            return "automatic"

        return "other"

    df["transmission_type"] = df["transmission"].apply(get_transmission_type)

    # 1-7. Remove rows missing essential variables.
    df = df.dropna(subset=["price", "milage", "model_year"])

    return df.reset_index(drop=True)


# ==========================================================================================
# 2. Missing Value Imputation After Split
# ==========================================================================================
def fill_missing_after_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill missing engine-related numeric values using train-set statistics only.

    Leakage prevention:
    - Brand-level medians are calculated from train_df only.
    - Global medians are calculated from train_df only.
    - The same train-derived rules are then applied to both train_df and test_df.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    num_cols = ["HP", "engine_L", "cylinders"]

    # 2-1. Electric vehicles do not have traditional engine displacement/cylinders.
    # Therefore, missing engine_L and cylinders are filled with 0 for Electric rows.
    for df_part in [train_df, test_df]:
        electric_mask = df_part["fuel_type"] == "Electric"
        df_part.loc[electric_mask, ["engine_L", "cylinders"]] = (
            df_part.loc[electric_mask, ["engine_L", "cylinders"]].fillna(0)
        )

    # 2-2. Calculate brand-level and global medians from train data only.
    brand_medians = train_df.groupby("brand")[num_cols].median()
    global_medians = train_df[num_cols].median()

    def apply_brand_median(df_part: pd.DataFrame) -> pd.DataFrame:
        df_part = df_part.copy()

        for col in num_cols:
            # First, use the matching brand median from train data.
            df_part[col] = df_part[col].fillna(df_part["brand"].map(brand_medians[col]))

            # If the brand is unseen or has no median, use global train median.
            df_part[col] = df_part[col].fillna(global_medians[col])

        return df_part

    train_df = apply_brand_median(train_df)
    test_df = apply_brand_median(test_df)

    # 2-3. Fill remaining missing transmission_type values.
    train_df["transmission_type"] = train_df["transmission_type"].fillna("unknown")
    test_df["transmission_type"] = test_df["transmission_type"].fillna("unknown")

    return train_df, test_df


# ==========================================================================================
# 3. Outlier Removal: Train Only
# ==========================================================================================
def remove_outliers_iqr_train_only(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove extreme outliers from the training set using the IQR rule.

    Important:
    - This function should be applied only to the training set.
    - The test set should remain unchanged to preserve realistic evaluation.
    """
    train_df = train_df.copy()

    for col in ["milage", "price"]:
        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        train_df = train_df[(train_df[col] >= lower) & (train_df[col] <= upper)]

    return train_df.reset_index(drop=True)


# ==========================================================================================
# 4. Classification Target Creation
# ==========================================================================================
def make_price_class_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Create price_class labels using train-set price quantiles.

    Classes:
    - low: price <= train 1/3 quantile
    - mid: train 1/3 quantile < price <= train 2/3 quantile
    - high: price > train 2/3 quantile

    Leakage prevention:
    - Cutoff values are calculated from train_df only.
    - The same cutoff values are applied to test_df.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    q1 = train_df["price"].quantile(1 / 3)
    q2 = train_df["price"].quantile(2 / 3)

    def classify_price(price: float) -> str:
        if price <= q1:
            return "low"
        if price <= q2:
            return "mid"
        return "high"

    train_df["price_class"] = train_df["price"].apply(classify_price)
    test_df["price_class"] = test_df["price"].apply(classify_price)

    thresholds = {
        "low_mid_threshold": float(q1),
        "mid_high_threshold": float(q2),
    }

    return train_df, test_df, thresholds


# ==========================================================================================
# 5. Convenience Helpers
# ==========================================================================================
def drop_high_cardinality_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop high-cardinality or already-transformed columns.

    Default dropped columns:
    - brand, model, engine, transmission, ext_col, int_col
    """
    if drop_cols is None:
        drop_cols = DROP_COLS

    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    return train_df, test_df


def prepare_classification_train_test(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Full preprocessing workflow for classification.

    Returns:
    - train_df: processed training dataframe with price_class
    - test_df: processed test dataframe with price_class
    - thresholds: train-derived low/mid/high cutoff values
    """
    df = pd.read_csv(csv_path)
    df_pre = preprocess_before_split(df)

    train_df, test_df = train_test_split(
        df_pre,
        test_size=test_size,
        random_state=random_state,
    )

    train_df, test_df = fill_missing_after_split(train_df, test_df)
    train_df, test_df, thresholds = make_price_class_train_test(train_df, test_df)
    train_df, test_df = drop_high_cardinality_columns(train_df, test_df)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), thresholds


def prepare_regression_train_test(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    remove_train_outliers: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing workflow for regression.

    Returns:
    - train_df: processed training dataframe
    - test_df: processed test dataframe

    Parameters:
    - remove_train_outliers:
        If True, IQR outlier removal is applied only to the train set.
        The default is False to match the uploaded regression script behavior.
    """
    df = pd.read_csv(csv_path)
    df_pre = preprocess_before_split(df)

    train_df, test_df = train_test_split(
        df_pre,
        test_size=test_size,
        random_state=random_state,
    )

    train_df, test_df = fill_missing_after_split(train_df, test_df)

    if remove_train_outliers:
        train_df = remove_outliers_iqr_train_only(train_df)

    train_df, test_df = drop_high_cardinality_columns(train_df, test_df)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# Backward-compatible alias used in the uploaded classification script.
prepare_train_test = prepare_classification_train_test
