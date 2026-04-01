from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_price_prediction.csv"

FEATURE_COLUMNS = [
    "year",
    "levy",
    "manufacturer",
    "fuel",
    "engine_volume",
    "mileage",
    "cylinders",
    "category",
]


class PricePredictor:
    def __init__(self) -> None:
        self.model = LinearRegression()
        self.manufacturer_map: dict[str, int] = {}
        self.fuel_map: dict[str, int] = {}
        self.manufacturer_choices: list[str] = []
        self.fuel_choices: list[str] = []
        self._fit()

    def _fit(self) -> None:
        df = pd.read_csv(DATA_PATH)

        # Match notebook column simplification.
        df = df.rename(
            columns={
                "ID": "id",
                "Price": "price",
                "Levy": "levy",
                "Manufacturer": "manufacturer",
                "Model": "model",
                "Prod. year": "year",
                "Category": "category",
                "Leather interior": "leather_interior",
                "Fuel type": "fuel",
                "Engine volume": "engine_volume",
                "Mileage": "mileage",
                "Cylinders": "cylinders",
                "Gear box type": "gear_type",
                "Drive wheels": "drive_wheels",
                "Doors": "doors",
                "Wheel": "wheel",
                "Color": "color",
                "Airbags": "airbags",
            }
        )

        # Match notebook filtering and cleaning.
        df["levy"] = df["levy"].replace("-", 0)
        df["levy"] = pd.to_numeric(df["levy"], errors="coerce")

        df = df[(df["price"] > 100) & (df["price"] < 600000) & (df["levy"] > 0)]

        df["mileage"] = (
            df["mileage"]
            .astype(str)
            .str.replace("km", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

        df["engine_volume"] = (
            df["engine_volume"]
            .astype(str)
            .str.replace("Turbo", "", regex=False)
            .str.strip()
        )
        df["engine_volume"] = pd.to_numeric(df["engine_volume"], errors="coerce")

        manufacturer_sorted = (
            df.groupby("manufacturer")["price"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
        self.manufacturer_map = {
            name: code for code, name in enumerate(manufacturer_sorted, start=1)
        }

        fuel_sorted = (
            df.groupby("fuel")["price"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
        self.fuel_map = {name: code for code, name in enumerate(fuel_sorted, start=1)}

        df["manufacturer"] = df["manufacturer"].map(self.manufacturer_map)
        df["fuel"] = df["fuel"].map(self.fuel_map)

        X = df[FEATURE_COLUMNS].copy()
        y = df["price"].copy()

        model_df = pd.concat([X, y], axis=1).dropna()
        X = model_df[FEATURE_COLUMNS]
        y = model_df["price"]

        self.model.fit(X, y)

        self.manufacturer_choices = manufacturer_sorted
        self.fuel_choices = fuel_sorted

    def predict(self, payload: dict[str, str]) -> float:
        manufacturer_code = self.manufacturer_map[payload["manufacturer"]]
        fuel_code = self.fuel_map[payload["fuel"]]

        row = pd.DataFrame(
            [
                {
                    "year": float(payload["year"]),
                    "levy": float(payload["levy"]),
                    "manufacturer": float(manufacturer_code),
                    "fuel": float(fuel_code),
                    "engine_volume": float(payload["engine_volume"]),
                    "mileage": float(payload["mileage"]),
                    "cylinders": float(payload["cylinders"]),
                }
            ]
        )

        pred = self.model.predict(row)[0]
        return float(max(pred, 0.0))


app = Flask(__name__)
predictor = PricePredictor()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    default_values = {
        "year": "2015",
        "levy": "1000",
        "manufacturer": predictor.manufacturer_choices[0],
        "fuel": predictor.fuel_choices[0],
        "engine_volume": "2.0",
        "mileage": "80000",
        "cylinders": "4",
    }

    form_values = default_values.copy()

    if request.method == "POST":
        form_values.update({k: request.form.get(k, "").strip() for k in default_values})
        try:
            prediction = predictor.predict(form_values)
        except (ValueError, KeyError):
            error = "Please enter valid values for all fields."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_values=form_values,
        manufacturers=predictor.manufacturer_choices,
        fuels=predictor.fuel_choices,
    )


if __name__ == "__main__":
    app.run(debug=True)
