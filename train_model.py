import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)


# =========================================================
# CUSTOMER LIFETIME VALUE - MODEL TRAINING
# =========================================================

print("=" * 60)
print("CUSTOMER LIFETIME VALUE - MODEL TRAINING")
print("=" * 60)


# =========================================================
# 1. LOAD PROCESSED DATA
# =========================================================

data_path = Path("data/processed/customer_data_processed.csv")

df = pd.read_csv(data_path)

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# =========================================================
# 2. DEFINE FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["CustomerLifetimeValue", "Date"])

y = df["CustomerLifetimeValue"]


# =========================================================
# 3. DEFINE CATEGORICAL AND NUMERICAL FEATURES
# =========================================================

categorical_features = [
    "Month",
    "Season",
    "Gender",
    "Region",
    "Membership",
    "DiscountUsed",
    "MarketingChannel",
    "ChurnRisk"
]

numerical_features = [
    "Age",
    "Income",
    "ProductsPurchased",
    "PurchaseFrequency",
    "AverageOrderValue",
    "SatisfactionScore"
]


# =========================================================
# 4. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# =========================================================
# 5. SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 6. DEFINE MODELS
# =========================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=8
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=12
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# =========================================================
# 7. TRAIN AND EVALUATE MODELS
# =========================================================

results = []

best_model = None
best_model_name = None
best_r2 = float("-inf")


for name, model in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Evaluation
    r2 = r2_score(y_test, predictions)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = root_mean_squared_error(
        y_test,
        predictions
    )

    print("R² Score :", round(r2, 4))
    print("MAE      :", round(mae, 2))
    print("RMSE     :", round(rmse, 2))

    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })

    # Find best model
    if r2 > best_r2:

        best_r2 = r2
        best_model = pipeline
        best_model_name = name


# =========================================================
# 8. CREATE MODELS DIRECTORY
# =========================================================

Path("models").mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 9. SAVE BEST MODEL
# =========================================================

model_path = "models/best_ltv_model.pkl"

joblib.dump(
    best_model,
    model_path
)


# =========================================================
# 10. SAVE MODEL RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

results_df.to_csv(
    "models/model_results.csv",
    index=False
)


# =========================================================
# 11. DISPLAY FINAL RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_name)
print("R² Score:", round(best_r2, 4))

print("\nBest model saved to:")
print(model_path)

print("\nModel comparison saved to:")
print("models/model_results.csv")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)