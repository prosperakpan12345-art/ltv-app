from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/best_ltv_model.pkl")

# Load model performance results
model_results_df = pd.read_csv("models/model_results.csv")

# Model performance results
model_results = model_results_df.to_dict("records")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            customer = pd.DataFrame([{
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
        