import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

rows = 5000

weather = np.random.choice(["Sunny", "Rainy", "Foggy"], rows)
road_type = np.random.choice(["Highway", "City", "Rural"], rows)
time_data = np.random.randint(0, 24, rows)
vehicle_count = np.random.randint(10, 250, rows)
speed = np.random.randint(20, 130, rows)

accident = []

for i in range(rows):
    risk = 0

    if weather[i] == "Rainy":
        risk += 2

    if weather[i] == "Foggy":
        risk += 3

    if speed[i] > 85:
        risk += 2

    if vehicle_count[i] > 150:
        risk += 2

    if time_data[i] in [8, 9, 10, 17, 18, 19]:
        risk += 2

    accident.append(1 if risk >= 5 else 0)

df = pd.DataFrame({
    "Weather": weather,
    "Road_Type": road_type,
    "Time": time_data,
    "Vehicle_Count": vehicle_count,
    "Speed": speed,
    "Accident": accident
})

df.loc[df.sample(frac=0.02).index, "Weather"] = np.nan
df.loc[df.sample(frac=0.02).index, "Speed"] = np.nan

df.to_csv("raw_traffic_data.csv", index=False)

df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])
df["Speed"] = df["Speed"].fillna(df["Speed"].median())

df.drop_duplicates(inplace=True)

df["Peak_Hour"] = df["Time"].apply(lambda x: 1 if x in [8, 9, 10, 17, 18, 19] else 0)
df["Day_Night"] = df["Time"].apply(lambda x: 1 if 6 <= x <= 18 else 0)
df["Traffic_Density"] = df["Vehicle_Count"] / (df["Speed"] + 1)

weather_map = {
    "Foggy": 0,
    "Rainy": 1,
    "Sunny": 2
}

road_map = {
    "City": 0,
    "Highway": 1,
    "Rural": 2
}

df["Weather"] = df["Weather"].map(weather_map)
df["Road_Type"] = df["Road_Type"].map(road_map)

scaler = StandardScaler()

scale_cols = [
    "Vehicle_Count",
    "Speed",
    "Traffic_Density"
]

df[scale_cols] = scaler.fit_transform(df[scale_cols])

df.to_csv("cleaned_traffic_data.csv", index=False)

plt.figure(figsize=(6, 4))
df["Accident"].value_counts().plot(kind="bar")
plt.title("Accident Distribution")
plt.xlabel("Accident")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("accident_distribution.png")
plt.close()

plt.figure(figsize=(6, 4))
df.groupby("Time")["Accident"].sum().plot(kind="line", marker="o")
plt.title("Traffic vs Time")
plt.xlabel("Hour")
plt.ylabel("Accidents")
plt.tight_layout()
plt.savefig("traffic_time_analysis.png")
plt.close()

plt.figure(figsize=(6, 4))
df.groupby("Weather")["Accident"].sum().plot(kind="bar")
plt.title("Weather Impact on Accidents")
plt.xlabel("Weather Encoded")
plt.ylabel("Accidents")
plt.tight_layout()
plt.savefig("weather_accident_analysis.png")
plt.close()

X = df.drop("Accident", axis=1)
y = df["Accident"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))

joblib.dump(model, "traffic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Project Complete")