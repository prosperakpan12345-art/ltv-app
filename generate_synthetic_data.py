"""Generate a reproducible synthetic Nigerian Naira customer dataset.

CustomerLifetimeValue is generated as:

    annual revenue
    * expected retained years
    * product breadth factor
    * discount realization factor
    * customer-level variation

Expected retained years depend on churn risk, membership tier, and satisfaction.
Income is an annual Naira amount and is intentionally not part of the target
formula; this lets the model learn whether income adds predictive value.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261004
ROW_COUNT = 1_000
OUTPUT_FILE = (
    Path(__file__).resolve().parent / "data" / "raw" / "customer_data_ngn.csv"
)

INCOME_RANGE = (500_000, 20_000_000)
AVERAGE_ORDER_VALUE_RANGE = (5_000, 400_000)
PURCHASE_FREQUENCY_RANGE = (1, 25)
PRODUCTS_PURCHASED_RANGE = (1, 50)

BASE_TENURE_YEARS = {
    "Low": 4.75,
    "Medium": 2.75,
    "High": 1.25,
}
MEMBERSHIP_TENURE_BONUS = {
    "Bronze": 0.0,
    "Silver": 0.25,
    "Gold": 0.5,
    "Platinum": 0.75,
}


def generate_customer_data(
    row_count: int = ROW_COUNT, seed: int = SEED
) -> pd.DataFrame:
    """Return synthetic customer profiles and formula-generated CLV targets."""
    if row_count < 3:
        raise ValueError("row_count must be at least 3 to cover range endpoints.")

    rng = np.random.default_rng(seed)
    income = np.rint(
        np.exp(
            rng.uniform(
                np.log(INCOME_RANGE[0]),
                np.log(INCOME_RANGE[1]),
                row_count,
            )
        )
    ).astype(int)
    average_order_value = np.round(
        np.exp(
            rng.uniform(
                np.log(AVERAGE_ORDER_VALUE_RANGE[0]),
                np.log(AVERAGE_ORDER_VALUE_RANGE[1]),
                row_count,
            )
        ),
        2,
    )
    purchase_frequency = rng.integers(
        PURCHASE_FREQUENCY_RANGE[0],
        PURCHASE_FREQUENCY_RANGE[1] + 1,
        row_count,
    )
    products_purchased = rng.integers(
        PRODUCTS_PURCHASED_RANGE[0],
        PRODUCTS_PURCHASED_RANGE[1] + 1,
        row_count,
    )

    income[:2] = INCOME_RANGE
    average_order_value[:2] = AVERAGE_ORDER_VALUE_RANGE
    purchase_frequency[:2] = PURCHASE_FREQUENCY_RANGE
    products_purchased[:2] = PRODUCTS_PURCHASED_RANGE

    membership = rng.choice(
        ["Bronze", "Silver", "Gold", "Platinum"],
        size=row_count,
        p=[0.4, 0.3, 0.2, 0.1],
    )
    satisfaction = np.round(rng.uniform(1.0, 5.0, row_count), 1)
    churn_score = (
        (3.0 - satisfaction) * 0.35
        - np.array([MEMBERSHIP_TENURE_BONUS[tier] for tier in membership]) * 0.5
        - (purchase_frequency - 13) * 0.025
        + rng.normal(0, 0.55, row_count)
    )
    churn_risk = np.where(
        churn_score < -0.35,
        "Low",
        np.where(churn_score < 0.55, "Medium", "High"),
    )
    discount_used = rng.choice(["Yes", "No"], size=row_count, p=[0.38, 0.62])

    tenure_years = np.array(
        [
            BASE_TENURE_YEARS[risk] + MEMBERSHIP_TENURE_BONUS[tier]
            for risk, tier in zip(churn_risk, membership)
        ],
        dtype=float,
    )
    tenure_years += (satisfaction - 3.0) * 0.12
    tenure_years += rng.normal(0, 0.12, row_count)
    tenure_years = np.maximum(0.6, tenure_years)

    product_breadth_factor = 0.9 + 0.1 * products_purchased / 50
    discount_realization_factor = np.where(discount_used == "Yes", 0.97, 1.0)
    customer_variation = np.clip(rng.normal(1.0, 0.04, row_count), 0.85, 1.15)
    lifetime_value = np.rint(
        average_order_value
        * purchase_frequency
        * tenure_years
        * product_breadth_factor
        * discount_realization_factor
        * customer_variation
    ).astype(int)

    start_date = pd.Timestamp("2024-01-01")
    dates = start_date + pd.to_timedelta(
        rng.integers(0, 1096, row_count), unit="D"
    )
    result = pd.DataFrame(
        {
            "CustomerID": np.arange(100_001, 100_001 + row_count),
            "Date": dates.strftime("%Y-%m-%d"),
            "Age": rng.integers(18, 76, row_count),
            "Gender": rng.choice(["Male", "Female"], size=row_count),
            "Region": rng.choice(
                [
                    "North Central",
                    "North East",
                    "North West",
                    "South East",
                    "South South",
                    "South West",
                ],
                size=row_count,
            ),
            "Income": income,
            "Membership": membership,
            "ProductsPurchased": products_purchased,
            "PurchaseFrequency": purchase_frequency,
            "AverageOrderValue": average_order_value,
            "DiscountUsed": discount_used,
            "MarketingChannel": rng.choice(
                ["Social Media", "Email", "Direct", "Online"],
                size=row_count,
            ),
            "SatisfactionScore": satisfaction,
            "ChurnRisk": churn_risk,
            "CustomerLifetimeValue": lifetime_value,
        }
    )
    return result


if __name__ == "__main__":
    data = generate_customer_data()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(data):,} synthetic customer records: {OUTPUT_FILE}")
    print(
        "CLV formula: annual AOV × purchase frequency × expected retained years "
        "× product breadth × discount realization × customer variation."
    )
    for column in (
        "Income",
        "AverageOrderValue",
        "PurchaseFrequency",
        "ProductsPurchased",
        "CustomerLifetimeValue",
    ):
        print(
            f"{column}: {data[column].min():,.2f}–"
            f"{data[column].max():,.2f}"
        )
