from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/best_ltv_model.pkl")


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
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)