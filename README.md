# Used Car Price Prediction

## 1. Project Overview

This project predicts used car prices and classifies vehicles into price levels using vehicle information.

We applied two machine learning approaches:

* **Regression:** Predict the actual used car price, `price`
* **Classification:** Predict the price category, `low`, `mid`, or `high`

The main purpose of this project is to compare how regression and classification can be used differently even though both approaches are based on the same original target variable, `price`.

---

## 2. Dataset

The dataset used in this project is the **Used Car Price Prediction Dataset** from Kaggle.

The original dataset contains:

* 4,009 rows
* 12 columns
* Numerical and categorical features
* Missing values
* Object-type numerical values
* Complex text columns such as `engine` and `transmission`

Main feature groups:

| Feature Group    | Columns                         |
| ---------------- | ------------------------------- |
| Basic Info       | brand, model, model_year        |
| Condition Info   | milage, accident, clean_title   |
| Performance Info | fuel_type, engine, transmission |
| Target           | price                           |

---

## 3. Business Objective

Used car prices are affected by various vehicle features such as model year, mileage, engine performance, and accident history.

The objectives of this project are:

1. Predict the actual used car price using regression.
2. Classify vehicles into low, mid, and high price levels using classification.

---

## 4. Data Preprocessing

The preprocessing process includes:

* Removing unnecessary spaces from column names and text values
* Converting `price` and `milage` from object type to numeric type
* Handling missing values based on the meaning of each column
* Extracting useful features from the `engine` column:

  * `HP`
  * `engine_L`
  * `cylinders`
  * `turbo`
* Simplifying `transmission` into `transmission_type`
* Applying train-based imputation to prevent data leakage
* Creating `price_class` using train-set quantiles

Important data leakage prevention rule:

> Fit preprocessing rules on the train set, then apply them to the test set.

---

## 5. Open Source SW Contribution

The main open-source-style contribution of this project is the implementation of reusable automatic model search functions.

### Main Functions

* `auto_regression_search()`
* `auto_classification_search()`

These functions automatically compare multiple combinations of:

* Scalers
* Encoders
* Machine learning models
* Hyperparameters
* Evaluation metrics

Each combination is evaluated using 5-Fold Cross Validation and an independent test set.

The functions return:

* Full result table
* Top 5 model combinations
* Best model combination

---

## 6. File Description

| File                            | Description                                                      |
| ------------------------------- | ---------------------------------------------------------------- |
| `preprocessing.py`              | Common preprocessing functions for regression and classification |
| `auto_regression_search.py`     | Automatic model search function for regression                   |
| `auto_classification_search.py` | Automatic model search function for classification               |
| `requirements.txt`              | Required Python libraries                                        |

---

## 7. Regression Modeling

### Target

* `price`

### Models

* RandomForestRegressor
* GradientBoostingRegressor

### Evaluation Metrics

* RMSE
* MAE
* R²

### Best Result

| Item       | Result                                            |
| ---------- | ------------------------------------------------- |
| Scaler     | MinMaxScaler                                      |
| Encoder    | OrdinalEncoder                                    |
| Model      | GradientBoostingRegressor                         |
| Parameters | n_estimators=200, learning_rate=0.05, max_depth=3 |
| RMSE       | 54,169                                            |
| MAE        | 15,042                                            |
| R²         | 0.475                                             |

---

## 8. Classification Modeling

### Target

* `price_class`

### Target Classes

| Class | Rule                    |
| ----- | ----------------------- |
| Low   | price ≤ 21,000          |
| Mid   | 21,000 < price ≤ 39,998 |
| High  | price > 39,998          |

### Models

* RandomForestClassifier
* GradientBoostingClassifier
* LogisticRegression

### Evaluation Metrics

* Accuracy
* Macro F1
* Weighted F1

### Best Result

| Item       | Result                                              |
| ---------- | --------------------------------------------------- |
| Scaler     | RobustScaler                                        |
| Encoder    | OrdinalEncoder                                      |
| Model      | RandomForestClassifier                              |
| Parameters | n_estimators=100, max_depth=20, min_samples_split=5 |
| Accuracy   | 82.09%                                              |
| Macro F1   | 81.43%                                              |

---

## 9. Regression vs Classification Comparison

| Criteria   | Regression                          | Classification            |
| ---------- | ----------------------------------- | ------------------------- |
| Target     | `price`                             | `price_class`             |
| Output     | Exact price                         | Low / Mid / High          |
| Strength   | Detailed price estimation           | Easy price-level decision |
| Limitation | Underestimated high-priced vehicles | Mid class confusion       |

Regression is useful when exact price estimation is needed, while classification is useful when quick price-level decision-making is needed.

---

## 10. How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the regression model search:

```bash
python auto_regression_search.py --csv used_cars.csv
```

Run the classification model search:

```bash
python auto_classification_search.py --csv used_cars.csv
```

If the dataset is stored in another folder, specify the path:

```bash
python auto_classification_search.py --csv data/used_cars.csv
```

---

## 11. Limitations

* High-priced vehicles were underestimated in the regression model.
* The mid class was more often confused with neighboring price levels in the classification model.
* Some high-cardinality features such as `brand` and `model` were removed to reduce model complexity.

---

## 12. Team

Team 6
Data Science Term Project
