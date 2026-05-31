"""
Used-car price-class classification code.

This module performs the classification part of the team project.
1. It creates the target label, price_class, from train-set price quantiles.
2. It trains models that directly predict low/mid/high price classes.

Important data leakage rule:
- Target variable: price_class.
- Input features: vehicle information only; both price and price_class are removed.
"""

from __future__ import annotations

# argparse was not used in the lab slides. It lets this file receive command-line
# options such as --csv, --outdir, --test-size, --random-state, --cv, and --skip-plots.
import argparse
# json is used only to save a parameter dictionary as a readable string in the CSV output.
import json
import platform
import re
# itertools.product creates every scaler x encoder x model-parameter combination.
from itertools import product
# pathlib.Path handles file and folder paths in a cleaner OS-independent way.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# clone creates a fresh, unfitted copy of a Scikit-learn estimator before each trial.
# This prevents one fitted model from carrying state into the next experiment.
from sklearn.base import clone
# ColumnTransformer applies different preprocessing to different column groups.
# Here, numeric columns go to a scaler and categorical columns go to an encoder.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
# make_scorer wraps f1_score so it can be used inside cross_validate.
    make_scorer,
)
# cross_validate was not directly used in the lecture examples. It is similar to
# cross_val_score but can calculate several evaluation metrics in one CV run.
from sklearn.model_selection import KFold, cross_validate, train_test_split
# Pipeline connects preprocessing and the model into one estimator. During CV, it
# fits scaling/encoding only on each training fold, which helps prevent leakage.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


CLASS_ORDER = ["low", "mid", "high"]
NUM_COLS = ["model_year", "milage", "HP", "engine_L", "cylinders"]
CAT_COLS = ["fuel_type", "accident", "clean_title", "turbo", "transmission_type"]
DROP_COLS = ["brand", "model", "engine", "transmission", "ext_col", "int_col"]

# ==========================================================================================
# [REGULAR EXPRESSION (REGEX) REFERENCE GUIDE]
# ==========================================================================================
r"""
Quick regex syntax cheat sheet for patterns used throughout this module:
  - \d      : Matches any single decimal digit (0-9).
  - +       : Multiplier; matches 1 or more repetitions of the preceding token (e.g., \d+).
  - \s      : Matches any whitespace character (spaces, tabs, line breaks).
  - * : Multiplier; matches 0 or more repetitions of the preceding token (optional match).
  - ?       : Quantifier; matches 0 or 1 of the preceding token (makes a token or group optional).
  - \b      : Asserts a word boundary position, ensuring keywords do not blend into surrounding letters.
  - [^0-9.] : Character set negation. Matches any single character that is NOT a digit or a dot.
  - (...)   : Capture Group. Instructs .extract() to retrieve only this portion of the string.
  - (?:...) : Non-Capturing Group. Groups text patterns logically without creating a separate output column.
2. Pandas String Functions Linked with Regex:
  - .str.replace(pattern, repl, regex=True) : Finds the pattern and replaces it (used for data cleaning).
  - .str.extract(pattern, flags=re.IGNORECASE)  : Extracts captured groups while ignoring case sensitivity.
  - .str.contains(pattern, case=False)          : Returns True/False if the pattern exists, ignoring case.
  """

# ==========================================================================================
# 1. Preprocessing Before Data Splitting
# ==========================================================================================
def preprocess_before_split(df):
    """
    Performs initial global data cleaning on the entire dataset before train/test splitting.
    Focuses on text normalization, basic structural filters, and regex specification parsing.
    """
    df = df.copy() # Avoid mutating the source dataframe by operating on a deep copy
    
    # 1-1. Structural String Trimming
    # Strip accidental leading/trailing spaces from column headers (e.g., " price " -> "price")
    df.columns = df.columns.str.strip()

    # Strip whitespaces from every cell containing text/object data types
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 1-2. Filtering Out Invalid Missing Placeholders
    # Drops rows where crucial spec entries were recorded with a special dash placeholder.
    df = df[df["fuel_type"] != "–"]
    df = df[df["engine"] != "–"]

    # 1-3. Basic Structural Imputation & Categorical Encoding Fixes
    df["fuel_type"] = df["fuel_type"].replace("not supported", np.nan) # Standardize text anomalies to NaN
    df["fuel_type"] = df["fuel_type"].fillna("Electric")               # Default missing fuels to Electric
    df["accident"] = df["accident"].fillna("Unknown")                 # Retain missing history under an explicit class
    df["clean_title"] = df["clean_title"].fillna("No")                 # Fallback missing titles conservatively to 'No'

    # 1-4. Cleaning Currency and Distance Fields
    # Pattern r"[^0-9.]" targets symbols like '$' and ',' so they can be replaced with an empty string ""
    df["price"] = (
        df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce") # Safely cast strings to numeric types; bad parses map to NaN

    # Clean the mileage tracking field using the identical numeric mask technique
    df["milage"] = (
        df["milage"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
    )
    df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

    # 1-5. Extracting Specifications via Pattern Matching
    engine = df["engine"].astype(str)
    
    # Pattern: Captures standalone integers or decimals that immediately precede the letters 'HP'
    df["HP"] = pd.to_numeric(
        engine.str.extract(r"(\d+(?:\.\d+)?)\s*HP", flags=re.IGNORECASE)[0],
        errors="coerce",
    )
    # Pattern: Tracks numeric values followed immediately by displacement metrics 'L' or 'Liter'
    df["engine_L"] = pd.to_numeric(
        engine.str.extract(r"(\d+(?:\.\d+)?)\s*(?:L|Liter)", flags=re.IGNORECASE)[0],
        errors="coerce",
    )

    # Parsing cylinder count via 3 fallback layouts due to chaotic string variations:
    # Pattern 1: Catches letters V, I, H, an optional hyphen, and isolates the 1-2 digit structure (e.g., V-6 -> 6)
    cyl1 = engine.str.extract(r"\b[VIH]\s*-?\s*(\d{1,2})\b", flags=re.IGNORECASE)[0]
    # Pattern 2: Captures digits leading into standard explicit text like '6 Cylinder'
    cyl2 = engine.str.extract(r"\b(\d{1,2})\s*Cylinder\b", flags=re.IGNORECASE)[0]
    # Pattern 3: Captures trailing numbers appended to layouts like 'Straight 6' or 'Flat 4'
    cyl3 = engine.str.extract(r"\b(?:Straight|Flat)\s*(\d{1,2})\b", flags=re.IGNORECASE)[0]
    # Sequentially merge extractions using .fillna() to compile a clean, unified cylinder column
    df["cylinders"] = pd.to_numeric(cyl1.fillna(cyl2).fillna(cyl3), errors="coerce")

    # Flag cars equipped with turbochargers by testing for the keyword "turbo" anywhere in the engine profile
    df["turbo"] = engine.str.contains("turbo", case=False, na=False)

    # 1-6. Standardizing Transmission Text Layouts
    def get_transmission_type(x):
        if pd.isna(x):
            return np.nan
        x = str(x).lower() # Downcase to unify multi-case data string checks
        if "cvt" in x:
            return "cvt"
        elif "dual" in x or "auto-shift" in x:
            return "dual_shift"
        elif "manual" in x or "m/t" in x:
            return "manual"
        elif "automatic" in x or "a/t" in x:
            return "automatic"
        else:
            return "other"

    df["transmission_type"] = df["transmission"].apply(get_transmission_type)
    
    # 1-7. Dropping Unusable Rows
    # Models cannot learn without a target (price) or fundamental physical coordinates (mileage, year).
    # Deletes rows missing any of these core variables, then reconstructs a sequential index mapping.
    df = df.dropna(subset=["price", "milage", "model_year"])
    return df.reset_index(drop=True)


# ==========================================================================================
# 2. Imputing Missing Values Post-Split (Data Leakage Prevention)
# ==========================================================================================
def fill_missing_after_split(train_df, test_df):
    """
    Imputes missing values using statistical metrics derived STRICTLY from the training set.
    This strict boundary completely prevents evaluation metrics from leaking test-set distribution properties.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    num_cols = ["HP", "engine_L", "cylinders"]

    # 2-1. Domain-Specific Imputation Rule for Electric Vehicles
    # Electric cars naturally do not possess mechanical engine displacements or engine cylinders.
    # We explicitly force NaN values to 0 for EVs to prevent general median stats from incorrectly adding engines to them.
    for df_part in [train_df, test_df]:
        electric_mask = df_part["fuel_type"] == "Electric"
        df_part.loc[electric_mask, ["engine_L", "cylinders"]] = (
            df_part.loc[electric_mask, ["engine_L", "cylinders"]].fillna(0)
        )

    # 2-2. Building Statistical Mapping Reference Matrices from Training Sets ONLY
    # Groups cars by manufacturer to calculate specific performance midpoints (e.g., Porsche median HP vs. Toyota median HP)
    brand_medians = train_df.groupby("brand")[num_cols].median()
    # Global backup baseline calculated across all rows to catch unrecognized brands present in the validation sets
    global_medians = train_df[num_cols].median()

    # Contextual replacement mapper
    def apply_brand_median(df_part):
        df_part = df_part.copy()
        for col in num_cols:
            # Step 1: Attempt to fill NaNs using the selected car's manufacturer median signature
            df_part[col] = df_part[col].fillna(df_part["brand"].map(brand_medians[col]))
            # Step 2: Fall back to the absolute dataset median if manufacturer midpoints cannot be determined
            df_part[col] = df_part[col].fillna(global_medians[col])
        return df_part

    # Apply the training-derived mapping configurations to both splits uniformly
    train_df = apply_brand_median(train_df)
    test_df = apply_brand_median(test_df)
    
    # 2-3. Categorical Restructuring
    # Catch any outstanding transmission NaNs and preserve them safely within a standard string label
    train_df["transmission_type"] = train_df["transmission_type"].fillna("unknown")
    test_df["transmission_type"] = test_df["transmission_type"].fillna("unknown")
    return train_df, test_df


# ==========================================================================================
# 3. Outlier Removal via Interquartile Range (IQR) - Training Data Only
# ==========================================================================================
def remove_outliers_iqr_train_only(train_df):
    """
    Removes extreme anomalies from key continuous features using the statistical IQR rule.
    CRITICAL: This must ONLY be applied to the training set. Testing datasets must retain their original 
    shape to ensure evaluation scores mimic authentic, unfiltered real-world scenarios.
    """
    train_df = train_df.copy()
    for col in ["milage", "price"]:
        # Calculate percentiles (Q1 = 25th percentile, Q3 = 75th percentile)
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1 # The range covering the middle 50% of your data
        
        # Establish dynamic boundaries using the standard 1.5 * IQR outlier formula
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Filter out rows falling outside the valid boundary range bounds
        train_df = train_df[(train_df[col] >= lower) & (train_df[col] <= upper)]
        
    return train_df.reset_index(drop=True)


# ==========================================================================================
# 4. Categorical Binning & Target Variable Discretization
# ==========================================================================================
def make_price_class_train_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Discretizes continuous numerical price records into uniform categorical tiers ('low', 'mid', 'high').
    Used to set up target ground truths for classification model configurations.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # [Prevent Data Leakage] Percentile boundaries MUST be calculated solely on the training data distribution.
    # The .quantile() markers ensure that the class distributions are perfectly balanced into exact 33.3% tranches.
    q1 = train_df["price"].quantile(1 / 3)  # Cutoff separating cheap cars from mid-tier options
    q2 = train_df["price"].quantile(2 / 3)  # Cutoff separating mid-tier options from high-end premium cars

    # Boundary condition router
    def classify_price(price):
        if price <= q1:
            return "low"
        elif price <= q2:
            return "mid"
        else:
            return "high"

    # Convert continuous price columns to discrete target classes for both frames using the training-derived cutoff rules
    train_df["price_class"] = train_df["price"].apply(classify_price)
    test_df["price_class"] = test_df["price"].apply(classify_price)
    
    # Return transformed subsets alongside the calculated cutting limits for final analytical summary rendering
    return train_df, test_df, {
        "low_mid_threshold": float(q1),
        "mid_high_threshold": float(q2),
    }


def prepare_train_test(
    csv_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Builds the train/test dataframes used by the classification experiment.

    Parameters:
    - csv_path: input CSV file path.
    - test_size: fraction of rows assigned to the independent test set.
    - random_state: fixed random seed so the split can be reproduced.
    """
    df = pd.read_csv(csv_path)
    df_pre = preprocess_before_split(df)

    # train_test_split separates rows into training and independent test sets.
    # test_size=0.2 means 20% of rows are held out for final testing.
    # random_state fixes the pseudo-random split for reproducibility.
    train_df, test_df = train_test_split(
        df_pre, test_size=test_size, random_state=random_state
    )
    train_df, test_df = fill_missing_after_split(train_df, test_df)

    train_df, test_df, thresholds = make_price_class_train_test(train_df, test_df)
    train_df = train_df.drop(columns=DROP_COLS)
    test_df = test_df.drop(columns=DROP_COLS)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), thresholds


def make_onehot_encoder() -> OneHotEncoder:
    """
    Creates OneHotEncoder in a way that works across Scikit-learn versions.

    Parameters:
    - handle_unknown="ignore": unseen test categories are encoded as all zeros
      instead of raising an error.
    - sparse_output=False / sparse=False: returns a dense NumPy array. Newer
      Scikit-learn uses sparse_output, while older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(scaler: object, encoder: object) -> ColumnTransformer:
    """
    Creates a ColumnTransformer for mixed numeric and categorical columns.

    ColumnTransformer was not explicitly shown in the provided lecture slides.
    It lets one pipeline scale NUM_COLS while separately encoding CAT_COLS.

    Parameters:
    - scaler: one of StandardScaler, MinMaxScaler, or RobustScaler.
    - encoder: one of OneHotEncoder or OrdinalEncoder.
    - transformers: list of (name, transformer, columns) tuples.
    """
    return ColumnTransformer(
        transformers=[
            ("num", scaler, NUM_COLS),
            ("cat", encoder, CAT_COLS),
        ]
    )


def split_classification_xy(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Uses price_class as y and removes both price and price_class from X."""
    y_train = train_df["price_class"]
    x_train = train_df.drop(columns=["price", "price_class"])
    y_test = test_df["price_class"]
    x_test = test_df.drop(columns=["price", "price_class"])
    return x_train, y_train, x_test, y_test


def auto_classification_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Searches scaler, encoder, model, and hyperparameter combinations.

    Parameters:
    - x_train, y_train: training features and labels.
    - x_test, y_test: final hold-out test features and labels.
    - cv: number of folds for k-fold cross validation. The default is 5.
    - random_state: seed used by models and KFold for reproducibility.
    """
    scalers = {
        # StandardScaler standardizes each numeric feature to mean 0 and variance 1.
        "Standard": StandardScaler(),
        # MinMaxScaler rescales each numeric feature into a fixed range, usually 0 to 1.
        "MinMax": MinMaxScaler(),
        # RobustScaler uses median and IQR, so it is less sensitive to outliers.
        "Robust": RobustScaler(),
    }
    encoders = {
        "OneHot": make_onehot_encoder(),
        # OrdinalEncoder parameters:
        # handle_unknown="use_encoded_value" prevents errors for unseen categories.
        # unknown_value=-1 assigns unseen categories a reserved integer code.
        "Ordinal": OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
    }
    models_with_params = {
        "RF": (
            # RandomForestClassifier is covered by the ensemble-learning lecture.
            # random_state fixes randomness; n_jobs=-1 uses all available CPU cores.
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                # n_estimators: number of decision trees in the forest.
                "n_estimators": [100, 200],
                # max_depth: maximum depth of each tree. None means no fixed limit.
                "max_depth": [None, 10, 20],
                # min_samples_split: minimum samples required to split an internal node.
                "min_samples_split": [2, 5],
            },
        ),
        "GB": (
            # GradientBoostingClassifier is covered by the ensemble-learning lecture.
            # random_state fixes the random parts of model training.
            GradientBoostingClassifier(random_state=random_state),
            {
                # n_estimators: number of boosting stages.
                "n_estimators": [100, 200],
                # learning_rate: contribution size of each boosting stage.
                "learning_rate": [0.05, 0.1],
                # max_depth: maximum depth of each weak decision tree.
                "max_depth": [3, 5],
            },
        ),
        "Logistic": (
            # LogisticRegression is covered by the supervised-learning lecture.
            # max_iter=3000 gives the optimizer enough iterations to converge.
            LogisticRegression(max_iter=3000, random_state=random_state),
            # C is the inverse regularization strength. Larger C means weaker regularization.
            {"C": [0.1, 1.0, 10.0]},
        ),
    }

    # KFold parameters:
    # n_splits=cv sets the number of folds, shuffle=True randomizes row order
    # before splitting, and random_state makes the shuffled folds reproducible.
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        # average="macro" gives each class equal weight, which is useful when
        # comparing low/mid/high classes. zero_division=0 avoids warnings when
        # a model predicts no samples for a class.
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        # average="weighted" weights each class by its support in the dataset.
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    }

    rows: list[dict[str, object]] = []
    best: dict[str, object] = {"cv_f1_macro": -np.inf, "cv_accuracy": -np.inf}
    total_combos = 0

    for scaler_name, scaler in scalers.items():
        for encoder_name, encoder in encoders.items():
            preprocessor = make_preprocessor(scaler, encoder)

            for model_name, (model, grid_params) in models_with_params.items():
                keys = list(grid_params.keys())
                value_lists = [grid_params[key] for key in keys]
                param_combos = [dict(zip(keys, values)) for values in product(*value_lists)]

                for params in param_combos:
                    total_combos += 1
                    # clone() resets any learned state so each trial starts from a clean model.
                    model_clone = clone(model)
                    # set_params(**params) applies the current hyperparameter combination.
                    model_clone.set_params(**params)
                    # Pipeline keeps preprocessing and modeling together. This is important
                    # because cross_validate then refits the scaler/encoder inside each fold.
                    pipe = Pipeline(
                        [
                            ("prep", clone(preprocessor)),
                            ("model", model_clone),
                        ]
                    )

                    # cross_validate parameters:
                    # cv=kf uses the KFold object above, scoring computes multiple metrics,
                    # n_jobs=-1 uses all CPU cores, and return_train_score=False avoids
                    # saving train-fold scores that are not needed for this report.
                    cv_scores = cross_validate(
                        pipe,
                        x_train,
                        y_train,
                        cv=kf,
                        scoring=scoring,
                        n_jobs=-1,
                        return_train_score=False,
                    )

                    cv_accuracy = float(cv_scores["test_accuracy"].mean())
                    cv_f1_macro = float(cv_scores["test_f1_macro"].mean())
                    cv_f1_weighted = float(cv_scores["test_f1_weighted"].mean())

                    pipe.fit(x_train, y_train)
                    y_pred = pipe.predict(x_test)

                    # accuracy_score is the fraction of exactly correct predictions.
                    test_accuracy = accuracy_score(y_test, y_pred)
                    # f1_score combines precision and recall. macro F1 gives the
                    # low, mid, and high classes equal importance.
                    test_f1_macro = f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    )
                    # weighted F1 reflects the number of samples in each class.
                    test_f1_weighted = f1_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    )

                    row = {
                        "scaler": scaler_name,
                        "encoder": encoder_name,
                        "model": model_name,
                        # json.dumps converts the parameter dictionary into a stable
                        # text form so it can be saved cleanly inside a CSV cell.
                        "params": json.dumps(params, sort_keys=True),
                        "cv_accuracy": round(cv_accuracy, 4),
                        "cv_f1_macro": round(cv_f1_macro, 4),
                        "cv_f1_weighted": round(cv_f1_weighted, 4),
                        "test_accuracy": round(test_accuracy, 4),
                        "test_f1_macro": round(test_f1_macro, 4),
                        "test_f1_weighted": round(test_f1_weighted, 4),
                    }
                    rows.append(row)

                    is_better = (
                        cv_f1_macro > best["cv_f1_macro"]
                        or (
                            cv_f1_macro == best["cv_f1_macro"]
                            and cv_accuracy > best["cv_accuracy"]
                        )
                    )
                    if is_better:
                        best = {
                            "estimator": pipe,
                            "y_pred": y_pred,
                            "scaler": scaler_name,
                            "encoder": encoder_name,
                            "model": model_name,
                            "params": params,
                            "cv_accuracy": cv_accuracy,
                            "cv_f1_macro": cv_f1_macro,
                            "cv_f1_weighted": cv_f1_weighted,
                            "test_accuracy": test_accuracy,
                            "test_f1_macro": test_f1_macro,
                            "test_f1_weighted": test_f1_weighted,
                        }

    print(f"Total {total_combos} classification combinations searched.")
    results_df = pd.DataFrame(rows).sort_values(
        ["cv_f1_macro", "cv_accuracy"], ascending=[False, False]
    )
    results_df = results_df.reset_index(drop=True)
    top5_df = results_df.head(5).reset_index(drop=True)
    return results_df, top5_df, best


def setup_plot_style() -> None:
    """
    Sets a readable plotting font depending on the operating system.

    platform.system() returns the OS name. This is a standard-library helper
    used only to avoid broken characters in saved Matplotlib figures.
    """
    if platform.system() == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif platform.system() == "Darwin":
        plt.rc("font", family="AppleGothic")
    plt.rcParams["axes.unicode_minus"] = False


def save_confusion_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """
    Saves a confusion-matrix heatmap for the final test predictions.

    Parameters:
    - y_true: actual class labels.
    - y_pred: predicted class labels.
    - title: plot title.
    - save_path: output PNG path.
    """
    setup_plot_style()
    # confusion_matrix counts actual-vs-predicted class pairs in CLASS_ORDER.
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    plt.figure(figsize=(6, 5))
    # seaborn.heatmap displays the matrix as a color-coded table.
    # annot=True prints counts in cells, fmt="d" formats them as integers,
    # and cmap="Blues" selects the color palette.
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_importance_plot(best: dict[str, object], save_path: Path) -> None:
    """
    Saves the top feature-importance values for tree-based models.

    feature_importances_ is a Scikit-learn attribute used by tree ensemble
    models such as Random Forest and Gradient Boosting.
    """
    estimator = best["estimator"]
    model = estimator.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    prep = estimator.named_steps["prep"]
    cat_transformer = prep.named_transformers_["cat"]
    # get_feature_names_out returns generated encoded-column names, especially
    # useful when OneHotEncoder expands one categorical column into many columns.
    if hasattr(cat_transformer, "get_feature_names_out"):
        cat_feature_names = list(cat_transformer.get_feature_names_out(CAT_COLS))
    else:
        cat_feature_names = CAT_COLS

    feature_names = NUM_COLS + cat_feature_names
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    setup_plot_style()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15), x="importance", y="feature")
    plt.title(f"Feature Importance Top 15 ({best['model']})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_text_report(
    output_path: Path,
    thresholds: dict[str, float],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_best: dict[str, object],
    y_test: pd.Series,
) -> None:
    """
    Writes the main text result file for the classification experiment.

    Parameters:
    - output_path: report file path.
    - thresholds: train-set price cutoffs for low/mid/high labels.
    - train_df, test_df: processed dataframes with price_class labels.
    - class_best: dictionary containing the selected best model and metrics.
    - y_test: actual labels for the independent test set.
    """
    class_pred = class_best["y_pred"]

    lines = [
        "Used Car Price Classification Report",
        "=" * 41,
        "",
        "Label definition",
        f"- low: price <= {thresholds['low_mid_threshold']:.2f}",
        (
            "- mid: "
            f"{thresholds['low_mid_threshold']:.2f} < price <= "
            f"{thresholds['mid_high_threshold']:.2f}"
        ),
        f"- high: price > {thresholds['mid_high_threshold']:.2f}",
        "",
        "Class distribution",
        "[train]",
        train_df["price_class"].value_counts().reindex(CLASS_ORDER).to_string(),
        "",
        "[test]",
        test_df["price_class"].value_counts().reindex(CLASS_ORDER).to_string(),
        "",
        "Best classification-only model",
        f"- scaler: {class_best['scaler']}",
        f"- encoder: {class_best['encoder']}",
        f"- model: {class_best['model']}",
        f"- params: {class_best['params']}",
        f"- CV accuracy: {class_best['cv_accuracy']:.4f}",
        f"- CV macro F1: {class_best['cv_f1_macro']:.4f}",
        f"- Test accuracy: {class_best['test_accuracy']:.4f}",
        f"- Test macro F1: {class_best['test_f1_macro']:.4f}",
        "",
        "Classification-only test report",
        # classification_report summarizes precision, recall, F1-score, and support
        # for each class. labels=CLASS_ORDER fixes the display order.
        classification_report(
            y_test,
            class_pred,
            labels=CLASS_ORDER,
            zero_division=0,
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """
    Defines command-line options for running this file.

    argparse parameters:
    - description: short text shown in the help message.
    - default: value used when the user does not pass the option.
    - type: converts the command-line string to a Python type.
    - action="store_true": stores True only when the flag is present.
    """
    parser = argparse.ArgumentParser(description="Used-car price-class classification")
    parser.add_argument("--csv", default="used_cars.csv", help="Path to used_cars.csv")
    parser.add_argument("--outdir", default="classification_outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Runs the full classification workflow from CSV loading to output saving."""
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            "Put used_cars.csv in this folder or pass --csv path/to/used_cars.csv."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, thresholds = prepare_train_test(
        csv_path=csv_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    x_train, y_train, x_test, y_test = split_classification_xy(train_df, test_df)

    print("[Data]")
    print(f"X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"X_test : {x_test.shape}, y_test : {y_test.shape}")
    print(f"Thresholds: {thresholds}")
    print()

    print("[Classification-only search]")
    results_df, top5_df, best = auto_classification_search(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=args.cv,
        random_state=args.random_state,
    )

    print("\n[Classification-only Top 10]")
    print(results_df.head(10).to_string(index=False))

    print("\n[Best Classification-only]")
    print(f"  Scaler       : {best['scaler']}")
    print(f"  Encoder      : {best['encoder']}")
    print(f"  Model        : {best['model']}")
    print(f"  Params       : {best['params']}")
    print(f"  CV Macro F1  : {best['cv_f1_macro']:.4f}")
    print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
    print(f"  Test Macro F1: {best['test_f1_macro']:.4f}")

    results_df.to_csv(outdir / "classification_results_all.csv", index=False)
    top5_df.to_csv(outdir / "classification_results_top5.csv", index=False)

    write_text_report(
        output_path=outdir / "classification_report.txt",
        thresholds=thresholds,
        train_df=train_df,
        test_df=test_df,
        class_best=best,
        y_test=y_test,
    )

    if not args.skip_plots:
        save_confusion_plot(
            y_test,
            best["y_pred"],
            "Classification-only Confusion Matrix",
            outdir / "classification_only_confusion_matrix.png",
        )
        save_feature_importance_plot(
            best, outdir / "classification_feature_importance.png"
        )

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
