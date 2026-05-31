"""
Used Car Price Regression Prediction
=========================================
- Preprocessing
- Automatic Exploration of Scaler × Encoder × Model × Hyperparameter (Top-level Single Function)
- K-Fold CV + Test Evaluation (RMSE / MAE / R²)
- Top 5 Comparison + Visualization (Pred vs Actual, Residual, Model Comparison, Feature Importance)
"""

import pandas as pd
import numpy as np
import re
import platform
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================================================================
# [REGULAR EXPRESSION (REGEX) REFERENCE GUIDE]
# ==========================================================================================
r"""
💡 Quick regex syntax cheat sheet for patterns used throughout this module:
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
    # Drops rows where crucial spec entries were recorded with a corrupt dash string '–'
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
# 3. Categorical Binning & Target Variable Discretization
# ==========================================================================================
def make_price_class_train_test(train_df, test_df):
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
    return train_df, test_df

# ==========================================================================================
# Column type separation
# ColumnTransformer applies the RIGHT preprocessing to the RIGHT column type: scaling for numeric features, encoding for categorical features. Applying one uniform transform to both types would destroy the original meaning of the data.
# ==========================================================================================
NUM_COLS = ["model_year", "milage", "HP", "engine_L", "cylinders"]
CAT_COLS = ["fuel_type", "accident", "clean_title", "turbo", "transmission_type"]
 
 
# ==========================================================================================
# 9. Automatic Search Function (Open Source SW Contribution)
# ==========================================================================================
def auto_regression_search(
    X_train, y_train, X_test, y_test,
    scalers, encoders, models_with_params,
    cv=5, random_state=42,
):
    """
    Automatically search every combination of Scaler x Encoder x Model x Hyperparam,
    recording both K-Fold CV and Test metrics (RMSE/MAE/R2) at the same time.
 
    Search space size (per the project spec):
      Scaler (3) x Encoder (2) x (RF 27 + GBR 8) = 210 combinations, each run with 5-fold CV.
 
    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data
    X_test, y_test : pd.DataFrame, pd.Series
        Test data
    scalers : dict[str, transformer]
        e.g. {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
    encoders : dict[str, transformer]
        e.g. {"OneHot": OneHotEncoder(handle_unknown='ignore', sparse_output=False)}
    models_with_params : dict[str, tuple(estimator, param_grid)]
        e.g. {"RF": (RandomForestRegressor(random_state=42),
                    {"n_estimators": [50, 100], "max_depth": [None, 10]})}
    cv : int, default=5
        Number of K-Fold splits
    random_state : int, default=42
 
    Returns
    -------
    results_df : pd.DataFrame
        cv/test scores for every combination (sorted ascending by test_rmse)
    top5_df : pd.DataFrame
        Top 5 combinations
    best : dict
        Best combination info. {"estimator", "y_pred", "scaler", "encoder",
        "model", "params", "test_rmse", "test_mae", "test_r2"}
 
    Examples
    --------
    >>> scalers = {"Standard": StandardScaler()}
    >>> encoders = {"OneHot": OneHotEncoder(handle_unknown='ignore', sparse_output=False)}
    >>> models = {"RF": (RandomForestRegressor(random_state=42),
    ...                  {"n_estimators": [100], "max_depth": [None, 10]})}
    >>> res, top5, best = auto_regression_search(
    ...     X_tr, y_tr, X_te, y_te, scalers, encoders, models, cv=5)
    """
    # KFold: split train data into 5 folds. random_state is fixed so every combination is evaluated on the exact same splits (fair comparison); shuffle=True removes bias from any pre-existing row ordering.
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    best = {"test_rmse": np.inf}
    total_combos = 0
 
    for s_name, scaler in scalers.items():
        for e_name, encoder in encoders.items():
            # ColumnTransformer: route numeric cols to the scaler and categorical cols to the encoder within a single fit/transform interface.
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", scaler, NUM_COLS),
                    ("cat", encoder, CAT_COLS),
                ]
            )
            for m_name, (model, grid_params) in models_with_params.items():
                # Generate all hyperparameter combinations from the grid via Cartesian product
                if grid_params:
                    keys = list(grid_params.keys())
                    value_lists = [grid_params[k] for k in keys]
                    combos = [dict(zip(keys, v)) for v in product(*value_lists)]
                else:
                    combos = [{}]
 
                for params in combos:
                    total_combos += 1
                    # clone() gives a fresh, unfitted copy so each combination starts clean
                    m_clone = clone(model)
                    if params:
                        m_clone.set_params(**params)
 
                    # Pipeline: chain preprocessing + model into one estimator.
                    # Crucially, when passed to cross_validate, the scaler/encoder are fit ONLY on each fold's train portion and merely transform the validation portion -> no leakage.
                    pipe = Pipeline([
                        ("prep", clone(preprocessor)),
                        ("model", m_clone),
                    ])
 
                    # ---- K-Fold CV: record RMSE/MAE/R2 simultaneously ----
                    # scikit-learn treats "higher score = better", so error metrics use the neg_prefix and come back negative; we flip the sign below to report positive values.
                    cv_scores = cross_validate(
                        pipe, X_train, y_train, cv=kf,
                        scoring={
                            "rmse": "neg_root_mean_squared_error",
                            "mae": "neg_mean_absolute_error",
                            "r2": "r2",
                        },
                        n_jobs=-1, return_train_score=False,
                    )
                    cv_rmse = -cv_scores["test_rmse"].mean()  # flip sign back to positive
                    cv_mae = -cv_scores["test_mae"].mean()
                    cv_r2 = cv_scores["test_r2"].mean()
 
                    # ---- Train Fit & Test evaluation ----
                    # Refit on the FULL training set, then measure real-world performance on the untouched test set. CV score ~ Test score => the model is trustworthy.
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    test_mae = mean_absolute_error(y_test, y_pred)
                    test_r2 = r2_score(y_test, y_pred)
 
                    rows.append({
                        "scaler": s_name,
                        "encoder": e_name,
                        "model": m_name,
                        "params": str(params),
                        "cv_rmse": round(cv_rmse, 2),
                        "cv_mae": round(cv_mae, 2),
                        "cv_r2": round(cv_r2, 4),
                        "test_rmse": round(test_rmse, 2),
                        "test_mae": round(test_mae, 2),
                        "test_r2": round(test_r2, 4),
                    })
 
                    # Track the single best combination by lowest test_rmse: RMSE is chosen as the primary criterion because it penalizes large errors heavily — costly mispredictions on expensive cars are the biggest business risk.
                    if test_rmse < best["test_rmse"]:
                        best = {
                            "estimator": pipe,
                            "y_pred": y_pred,
                            "scaler": s_name,
                            "encoder": e_name,
                            "model": m_name,
                            "params": params,
                            "test_rmse": test_rmse,
                            "test_mae": test_mae,
                            "test_r2": test_r2,
                        }
 
    print(f"\nTotal {total_combos} of combination searched.")
    # Sort all results ascending by test_rmse and slice the top 5 
    results_df = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    top5_df = results_df.head(5).reset_index(drop=True)
    return results_df, top5_df, best
 
 
# ==========================================================================================
# 12. Visualization
# Four presentation-ready plots are generated automatically to communicate results.
# ==========================================================================================
def setup_korean_font():
    """Configure a CJK-capable font so Korean labels (if any) render without tofu boxes."""
    if platform.system() == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif platform.system() == "Darwin":
        plt.rc("font", family="AppleGothic")
    plt.rcParams["axes.unicode_minus"] = False  # Render minus signs correctly with CJK fonts
 
 
def plot_predicted_vs_actual(y_test, y_pred, save_path=None):
    """12-1. Predicted vs Actual: the farther points stray from the red diagonal, the larger the error."""
    setup_korean_font()
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.4)
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lim, lim, "r--", label="Perfect prediction")  # y = x reference line
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual Price")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_residual(y_test, y_pred, save_path=None):
    """12-2. Residual plot: an ideal model scatters residuals evenly around 0; skew reveals model limits."""
    setup_korean_font()
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.4)
    plt.axhline(0, color="r", linestyle="--")  # Zero-error reference line
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_model_comparison(results_df, save_path=None):
    """12-3. Model x Scaler bar chart: compare at a glance which model/scaler combo performs best."""
    setup_korean_font()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="model", y="test_rmse", hue="scaler")
    plt.xticks(rotation=20)
    plt.title("Test RMSE: Model × Scaler Comparing")
    plt.xlabel("Model")
    plt.ylabel("Test RMSE")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_feature_importance(best, save_path=None):
    """12-4. Feature Importance (Top 15): identify the variables that drive price predictions most."""
    setup_korean_font()
    model = best["estimator"].named_steps["model"]
    # Linear models lack feature_importances_; only tree-based ensembles expose it.
    if not hasattr(model, "feature_importances_"):
        print(f"[Notice] {best['model']} Model doesn't support feature_importances.")
        return
 
    # Reconstruct the post-transform feature names. OneHotEncoder expands one categorical column into many, so we must pull names from the fitted encoder to align with importances.
    prep = best["estimator"].named_steps["prep"]
    cat_transformer = prep.named_transformers_["cat"]
    if hasattr(cat_transformer, "get_feature_names_out"):
        cat_feature_names = cat_transformer.get_feature_names_out(CAT_COLS)
    else:
        cat_feature_names = CAT_COLS
    feature_names = list(NUM_COLS) + list(cat_feature_names)
 
    importances = model.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )
 
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title(f"Feature Importance Top 15 (Best Model: {best['model']})", fontsize=14)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# Main
if __name__ == "__main__":
    # 1. Load data & preprocess
    df = pd.read_csv("used_cars.csv")
    df_pre = preprocess_before_split(df)
 
    # 2. Train / Test split
    train_df, test_df = train_test_split(df_pre, test_size=0.2, random_state=42)
 
    # 3. Handle missing values (after split - to prevent leakage)
    train_df, test_df = fill_missing_after_split(train_df, test_df)
 
    # (price_class is not needed for regression -> skip make_price_class_train_test)
 
    # 5. Drop unnecessary high-cardinality columns
    drop_cols = ["brand", "model", "engine", "transmission", "ext_col", "int_col"]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)
 
    # 6. Split X / y (regression: the target is the continuous 'price')
    y_train = train_df["price"]
    X_train = train_df.drop(columns=["price"])
    y_test = test_df["price"]
    X_test = test_df.drop(columns=["price"])
 
    print("\n[Data]")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")
 
    # ============================================
    # Define the search space
    # ============================================
    # Scaler candidates: the optimal scaler depends on the data distribution.
    #   Standard -> mean 0, std 1 | MinMax -> rescale into [0, 1] | Robust -> median/IQR (outlier-resistant)
    scalers = {
        "Standard": StandardScaler(),
        "MinMax": MinMaxScaler(),
        "Robust": RobustScaler(),
    }
    # Encoder candidates: handle_unknown keeps unseen test-set categories from erroring.
    #   OneHot -> one 0/1 column per category (safer for linear models)
    #   Ordinal -> map each category to a single integer (fine for tree models)
    encoders = {
        "OneHot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        "Ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    }
    # Model candidates: two tree-based ensembles.
    #   RandomForest    -> stable with many features, strong against overfitting (27 grid combos)
    #   GradientBoosting -> sequentially corrects prior trees' residuals, usually top accuracy (8 grid combos)
    models_with_params = {
        "RF": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "GBR": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
        # Additional model examples (uncomment if needed):
        # "Linear": (LinearRegression(), {}),
        # "Ridge":  (Ridge(), {"alpha": [0.1, 1, 10]}),
        # "Lasso":  (Lasso(max_iter=10000), {"alpha": [0.01, 0.1, 1]}),
    }
 
    # ============================================
    # Automatic search
    # ============================================
    print("\n[Auto Regression Search]")
    results_df, top5_df, best = auto_regression_search(
        X_train, y_train, X_test, y_test,
        scalers=scalers,
        encoders=encoders,
        models_with_params=models_with_params,
        cv=5,
        random_state=42,
    )
 
    # ============================================
    # Print results
    # ============================================
    print("\n[Total Result- Top 10]")
    print(results_df.head(10).to_string(index=False))
 
    print("\n[Top 5 Best Combinations]")
    print(top5_df.to_string(index=False))
 
    print("\n[Best Combination]")
    print(f"  Scaler   : {best['scaler']}")
    print(f"  Encoder  : {best['encoder']}")
    print(f"  Model    : {best['model']}")
    print(f"  Params   : {best['params']}")
    print(f"  Test RMSE: {best['test_rmse']:.2f}")
    print(f"  Test MAE : {best['test_mae']:.2f}")
    print(f"  Test R²  : {best['test_r2']:.4f}")
 
    # ============================================
    # Save results
    # Persist tables and plots so they can be dropped straight into a report without re-running code.
    # ============================================
    results_df.to_csv("results_all.csv", index=False)
    top5_df.to_csv("results_top5.csv", index=False)
    print("\nresults_all.csv, results_top5.csv")
 
    # ============================================
    # Visualization
    # ============================================
    plot_predicted_vs_actual(y_test, best["y_pred"], save_path="pred_vs_actual.png")
    plot_residual(y_test, best["y_pred"], save_path="residual.png")
    plot_model_comparison(results_df, save_path="model_comparison.png")
    plot_feature_importance(best, save_path="feature_importance.png")
    print("pred_vs_actual.png, residual.png, model_comparison.png, feature_importance.png")
    
    # ==========================================================================
    # [ADDED] Convert Continuous Regression Predictions (Price) into Classification Classes
    # ==========================================================================
    # Purpose: To fairly and quantitatively compare two different approaches:
    #          1) A classification model that directly predicts the price tier.
    #          2) A regression model that accurately predicts the exact price first, 
    #             which is then mapped into a corresponding price tier.
    
    print("\n" + "="*20 + " Regressor Predictions -> Classification Tiers Evaluation " + "="*20)
    
    # 1. [Prevent Data Leakage] Compute quantiles based ONLY on the 'Training Data (train_df)',
    #    exactly matching the logic used in the classification script.
    #    The .quantile() function identifies the values at specific percentiles (0 to 1).
    #    Here, we find the 33.3% (1/3) and 66.6% (2/3) marks to serve as thresholds.
    q1 = train_df["price"].quantile(1 / 3)  # Threshold separating 'low' and 'mid' tiers
    q2 = train_df["price"].quantile(2 / 3)  # Threshold separating 'mid' and 'high' tiers
    
    print(f"[Thresholds] low_mid: {q1:.2f} | mid_high: {q2:.2f}")

    # [User-Defined Function] Maps a continuous numerical price to its corresponding tier string based on q1 and q2.
    def convert_to_class(price):
        if price <= q1:
            return "low"        # Prices less than or equal to q1 are mapped to 'low'
        elif price <= q2:
            return "mid"        # Prices greater than q1 and less than or equal to q2 are mapped to 'mid'
        else:
            return "high"       # Prices greater than q2 are mapped to 'high'

    # 2. Convert both the ground truth prices (y_test) and the regressor's predictions (best["y_pred"]) into tier labels.
    CLASS_ORDER = ["low", "mid", "high"] # Explicitly define the label order for consistent reporting and matrix alignment.
    
    # Use pandas .apply() to map the conversion function over the entire series.
    y_test_class = y_test.apply(convert_to_class) # Ground truth prices -> Actual classes (pandas Series of strings)
    
    # Since best["y_pred"] is a NumPy array, wrap it in a pd.Series before using .apply().
    y_pred_class = pd.Series(best["y_pred"]).apply(convert_to_class) # Predicted prices -> Predicted classes (pandas Series of strings)

    # 3. Import classification evaluation metrics from scikit-learn.
    # - classification_report: Generates a text summary of Precision, Recall, F1-score, and Support.
    # - confusion_matrix: Computes a cross-tabulation matrix of actual vs. predicted classes to analyze error patterns.
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n[Regressor converted to Classification - Report]")
    # zero_division=0: Safe option that prevents execution errors by setting the score to 0 
    # if the model never predicts a certain class (which makes the denominator 0).
    print(classification_report(
        y_test_class, 
        y_pred_class, 
        labels=CLASS_ORDER, 
        zero_division=0
    ))
    
    print("[Regressor converted to Classification - Confusion Matrix]")
    # Compute the raw confusion matrix array using scikit-learn.
    cm = confusion_matrix(y_test_class, y_pred_class, labels=CLASS_ORDER)
    
    # Convert the raw NumPy array into a structured pandas DataFrame for enhanced readability in the console.
    # Assign clear prefixes to row indexes (Actual) and column headers (Predicted).
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in CLASS_ORDER], 
                             columns=[f"Pred_{c}" for c in CLASS_ORDER])
    print(cm_df) # Print the formatted confusion matrix to the terminal
    print("="*65)