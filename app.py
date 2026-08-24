from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/best_ltv_model.pkl")


# Model performance results
model_results = [
    {
        "model": "Gradient Boosting",
        "r2": 0.8722,
        "mae": 740.87,
        "rmse": 1253.85
    },
    {
        "model": "Random Forest",
        "r2": 0.8412,
        "mae": 843.05,
        "rmse": 1397.57
    },
    {
        "model": "Linear Regression",
        "r2": 0.8180,
        "mae": 1107.91,
        "rmse": 1496.40
    },
    {
        "model": "Decision Tree",
        "r2": 0.7770,
        "mae": 1060.11,
        "rmse": 1656.29
    }
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            # Get information from the form
            customer = pd.DataFrame([{
                "CustomerID": 9999,
                "Month": request.form["Month"],
                "Season": request.form["Season"],
                "Age": float(request.form["Age"]),
                "Gender": request.form["Gender"],
                "Region": request.form["Region"],
                "Income": float(request.form["Income"]),
                "Membership": request.form["Membership"],
                "ProductsPurchased": float(
                    request.form["ProductsPurchased"]
                ),
                "PurchaseFrequency": float(
                    request.form["PurchaseFrequency"]
                ),
                "AverageOrderValue": float(
                    request.form["AverageOrderValue"]
                ),
                "DiscountUsed": request.form["DiscountUsed"],
                "MarketingChannel": request.form["MarketingChannel"],
                "SatisfactionScore": float(
                    request.form["SatisfactionScore"]
                ),
                "ChurnRisk": request.form["ChurnRisk"]
            }])

            # Remove CustomerID because it was not used
            # as a prediction feature during training
            customer = customer.drop(columns=["CustomerID"])

            # Make prediction
            prediction = model.predict(customer)[0]

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        model_results=model_results
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
