import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "best_ltv_model.pkl")
model_results_df = pd.read_csv(BASE_DIR / "models" / "model_results.csv")
model_results = model_results_df.to_dict("records")
best_model = max(model_results, key=lambda result: result["R2"])

training_data = pd.read_csv(
    BASE_DIR / "data" / "processed" / "customer_data_processed.csv"
)
INPUT_RANGES = {
    field: {
        "min": float(training_data[column].min()),
        "max": float(training_data[column].max()),
    }
    for field, column in (
        ("Age", "Age"),
        ("Income", "Income"),
        ("ProductsPurchased", "ProductsPurchased"),
        ("PurchaseFrequency", "PurchaseFrequency"),
        ("AverageOrderValue", "AverageOrderValue"),
        ("SatisfactionScore", "SatisfactionScore"),
    )
}

NUMERIC_FIELDS = {
    "Age": "Age",
    "Income": "Annual income",
    "ProductsPurchased": "Products purchased",
    "PurchaseFrequency": "Purchases per year",
    "AverageOrderValue": "Average order value",
    "SatisfactionScore": "Satisfaction score",
}

QUARTER_BY_MONTH = {
    "January": "Q1",
    "February": "Q1",
    "March": "Q1",
    "April": "Q2",
    "May": "Q2",
    "June": "Q2",
    "July": "Q3",
    "August": "Q3",
    "September": "Q3",
    "October": "Q4",
    "November": "Q4",
    "December": "Q4",
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    selected_month = "January"
    selected_season = "Q1"
    form_values = {}

    if request.method == "POST":
        try:
            selected_month = request.form.get("Month", "").strip()
            selected_season = QUARTER_BY_MONTH.get(selected_month, "")

            if not selected_season:
                raise ValueError("Select a valid month.")

            def required_value(field):
                value = request.form.get(field, "").strip()
                if not value:
                    raise ValueError(f"{NUMERIC_FIELDS.get(field, field)} is required.")
                return value

            customer_values = {
                "Month": selected_month,
                "Season": selected_season,
                "Gender": required_value("Gender"),
                "Region": required_value("Region"),
                "Membership": required_value("Membership"),
                "DiscountUsed": required_value("DiscountUsed"),
                "MarketingChannel": required_value("MarketingChannel"),
                "ChurnRisk": required_value("ChurnRisk"),
            }

            for field, label in NUMERIC_FIELDS.items():
                raw_value = required_value(field)
                try:
                    value = float(raw_value)
                except ValueError:
                    raise ValueError(f"{label} must be a number.") from None
                limits = INPUT_RANGES[field]

                if not math.isfinite(value):
                    raise ValueError(f"{label} must be a finite number.")

                if not limits["min"] <= value <= limits["max"]:
                    unit = (
                        "₦"
                        if field in ("Income", "AverageOrderValue")
                        else ""
                    )
                    precision = (
                        2
                        if field in ("AverageOrderValue", "SatisfactionScore")
                        else 0
                    )
                    minimum = f"{limits['min']:,.{precision}f}"
                    maximum = f"{limits['max']:,.{precision}f}"
                    raise ValueError(
                        f"{label} must be between {unit}{minimum} and "
                        f"{unit}{maximum} "
                        "to stay within the training data range."
                    )

                customer_values[field] = value
                form_values[field] = raw_value

        except (KeyError, ValueError) as validation_error:
            error = str(validation_error)
        else:
            customer = pd.DataFrame([customer_values])
            prediction = model.predict(customer)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        model_results=model_results,
        best_model=best_model,
        input_ranges=INPUT_RANGES,
        form_values=form_values,
        selected_month=selected_month,
        selected_season=selected_season,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
