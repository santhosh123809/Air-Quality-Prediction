from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

DATA = Path("data/raw/city_day.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "AQI"]).copy()
df = df.drop_duplicates()

# Pollutants only: do not include AQI or AQI_Bucket
features = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene",
    "Toluene", "Xylene"
]

features = [c for c in features if c in df.columns]

if not features:
    raise ValueError("No expected pollutant columns found.")

for col in features + ["AQI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values("Date").reset_index(drop=True)

# Hold out the latest 20% of dates
dates = np.sort(df["Date"].dropna().unique())
cutoff = dates[int(len(dates) * 0.8)]

train = df[df["Date"] < cutoff].copy()
test = df[df["Date"] >= cutoff].copy()

X_train = train[features]
y_train = train["AQI"]
X_test = test[features]
y_test = test["AQI"]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

best_model = None
best_mae = float("inf")

for name, estimator in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.3f}")

    if mae < best_mae:
        best_mae = mae
        best_model = pipeline

joblib.dump(
    {"model": best_model, "features": features},
    OUT / "aqi_model.joblib"
)

# Save test predictions for analysis
results = test[["Date", "City", "AQI"]].copy()
results["Predicted_AQI"] = best_model.predict(X_test)
results["Absolute_Error"] = (
    results["AQI"] - results["Predicted_AQI"]
).abs()

results.to_csv(OUT / "test_predictions.csv", index=False)

print("\nBest model saved.")
print("Test predictions saved.")