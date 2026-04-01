from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


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
    "gear_type",
]


class PricePredictor:
    def __init__(self) -> None:
        self.model = RandomForestRegressor(random_state=71, n_estimators=101)
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

        # Clean and convert 'levy' column to numeric, treating '-' as 0.
        df["levy"] = df["levy"].replace("-", 0)
        df["levy"] = pd.to_numeric(df["levy"], errors="coerce")

        # Filter out outlier price and levy values.
        df = df[(df["price"] > 100) & (df["price"] < 600000)]

        # Clean and convert 'mileage' column to numeric by removing 'km' and commas.
        df["mileage"] = (
            df["mileage"]
            .astype(str)
            .str.replace("km", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

        # Clean and convert 'engine_volume' column to numeric by removing 'Turbo' and any non-numeric characters.
        df["engine_volume"] = (
            df["engine_volume"]
            .astype(str)
            .str.replace("Turbo", "", regex=False)
            .str.strip()
        )
        df["engine_volume"] = pd.to_numeric(df["engine_volume"], errors="coerce")

        # Create mappings for 'manufacturer' based on median price to preserve ordinal relationships.
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
        
        # Create mappings for 'category' based on median price to preserve ordinal relationships.
        category_sorted = (
            df.groupby('category')['price']
            .median()
            .sort_values()
            .index
        )
        self.category_map = {name: code for code, name in enumerate(category_sorted, start=1)}

        # Create mappings for 'fuel' based on median price to preserve ordinal relationships.
        fuel_sorted = (
            df.groupby("fuel")["price"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
        self.fuel_map = {name: code for code, name in enumerate(fuel_sorted, start=1)}

        # Create mappings for 'gear_type' 
        self.gear_type_map = {v: i for i, v in enumerate(sorted(df['gear_type'].dropna().unique()))}
        
        df['gear_type'] = df['gear_type'].map(self.gear_type_map)
        df["manufacturer"] = df["manufacturer"].map(self.manufacturer_map)
        df['category'] = df['category'].map(self.category_map)
        df["fuel"] = df["fuel"].map(self.fuel_map)

        x = df[FEATURE_COLUMNS].copy()
        y = df["price"].copy()

        model_df = pd.concat([x, y], axis=1).dropna()
        x = model_df[FEATURE_COLUMNS]
        y = model_df["price"]

        self.model.fit(x, y)

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
        except Exception as exc:
            app.logger.exception("Prediction failed with payload=%s", form_values)
            error = "Please enter valid values for all fields."
            if app.debug:
                error = f"{error} Debug: {exc}"

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
