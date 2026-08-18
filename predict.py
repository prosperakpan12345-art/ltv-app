import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/best_ltv_model.pkl")

# Example customer
customer = pd.DataFrame([{
    "CustomerID": 1001,
    "Date": "2026-08-13",
    "Month": "August",
    "Season": "Q3",
    "Age": 30,
    "Gender": "Female",
    "Region": "South",
    "Income": 80000,
    "Membership": "Gold",
    "ProductsPurchased": 10,
    "PurchaseFrequency": 7,
    "AverageOrderValue": 150,
    "DiscountUsed": "No",
    "MarketingChannel": "Social Media",
    "SatisfactionScore": 4.5,
    "ChurnRisk": "Low"
}])

# Remove fields that were not used for training
customer = customer.drop(columns=["CustomerID"])

# Make prediction
prediction = model.predict(customer)

print("=" * 50)
print("CUSTOMER LIFETIME VALUE PREDICTION")
print("=" * 50)

print("Predicted Customer Lifetime Value:")
print(round(prediction[0], 2))