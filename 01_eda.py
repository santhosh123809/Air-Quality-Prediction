from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = Path("data/raw/city_day.csv")
df = pd.read_csv(path)

# Standardize column names
df.columns = df.columns.str.strip()

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

df = df.dropna(subset=["Date", "City"])

# Remove exact duplicate records
df = df.drop_duplicates()

# Convert numeric columns
for col in df.columns:
    if col not in ["City", "Date", "AQI_Bucket"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

print("Dataset shape:", df.shape)
print("\nMissing values:\n")
print(df.isnull().sum().sort_values(ascending=False))
print("\nSummary:\n")
print(df.describe(include="all"))

# AQI trend over time across cities
city = "Chennai"
city_df = df[df["City"] == city].copy()

daily = (
    city_df.dropna(subset=["AQI"])
    .sort_values("Date")
)

plt.figure(figsize=(12, 5))
sns.lineplot(data=daily, x="Date", y="AQI")
plt.title(f"Historical AQI Trend - {city}")
plt.xlabel("Date")
plt.ylabel("Air Quality Index")
plt.tight_layout()
plt.show()

# Correlation between pollutants and AQI
numeric = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric.corr(),
    cmap="coolwarm",
    center=0
)
plt.title("Pollutant and AQI Correlation")
plt.tight_layout()
plt.show()