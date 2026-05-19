import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# PROJECT INFO
# =========================
print("AI Traffic & Accident Analytics")
print("Training Pipeline Started...")


# =========================
# CREATE SAMPLE RAW DATASET
# =========================
np.random.seed(42)

rows = 5000

weather = np.random.choice(
    ["Sunny", "Rainy", "Foggy"],
    rows
)

road_type = np.random.choice(
    ["Highway", "City", "Rural"],
    rows
)

time_data = np.random.randint(
    0,
    24,
    rows
)

vehicle_count = np.random.randint(
    10,
    250,
    rows
)

speed = np.random.randint(
    20,
    130,
    rows
)

accident = []

for i in range(rows):
    risk_score = 0

    if weather[i] == "Rainy":
        risk_score += 25

    if weather[i] == "Foggy":
        risk_score += 35

    if vehicle_count[i] > 150:
        risk_score += 25

    elif vehicle_count[i] > 90:
        risk_score += 15

    if speed[i] > 90:
        risk_score += 25

    elif speed[i] > 60:
        risk_score += 15

    if time_data[i] in [8, 9, 10, 17, 18, 19]:
        risk_score += 15

    if risk_score >= 70:
        accident.append(1)
    else:
        accident.append(0)


raw_df = pd.DataFrame({
    "Weather": weather,
    "Road_Type": road_type,
    "Time": time_data,
    "Vehicle_Count": vehicle_count,
    "Speed": speed,
    "Accident": accident
})


# =========================
# ADD REALISTIC DATA ISSUES
# =========================
raw_df.loc[raw_df.sample(frac=0.02, random_state=1).index, "Weather"] = np.nan
raw_df.loc[raw_df.sample(frac=0.02, random_state=2).index, "Road_Type"] = np.nan
raw_df.loc[raw_df.sample(frac=0.02, random_state=3).index, "Speed"] = np.nan

duplicate_rows = raw_df.sample(
    n=50,
    random_state=4
)

raw_df = pd.concat(
    [
        raw_df,
        duplicate_rows
    ],
    ignore_index=True
)

raw_df.to_csv(
    "raw_traffic_data.csv",
    index=False
)

print("Raw dataset created: raw_traffic_data.csv")


# =========================
# DATA PREPROCESSING
# =========================
df = raw_df.copy()

print("Preprocessing Started...")


# =========================
# MISSING VALUE HANDLING
# =========================
df["Weather"] = df["Weather"].fillna(
    df["Weather"].mode()[0]
)

df["Road_Type"] = df["Road_Type"].fillna(
    df["Road_Type"].mode()[0]
)

df["Time"] = df["Time"].fillna(
    df["Time"].median()
)

df["Vehicle_Count"] = df["Vehicle_Count"].fillna(
    df["Vehicle_Count"].median()
)

df["Speed"] = df["Speed"].fillna(
    df["Speed"].median()
)

df["Accident"] = df["Accident"].fillna(
    df["Accident"].mode()[0]
)


# =========================
# DUPLICATE REMOVAL
# =========================
df.drop_duplicates(
    inplace=True
)


# =========================
# STANDARDIZATION
# =========================
df["Weather"] = df["Weather"].astype(str).str.strip().str.title()
df["Road_Type"] = df["Road_Type"].astype(str).str.strip().str.title()


# =========================
# FEATURE ENGINEERING
# =========================
df["Peak_Hour"] = df["Time"].apply(
    lambda x: 1 if int(x) in [8, 9, 10, 17, 18, 19] else 0
)

df["Day_Night"] = df["Time"].apply(
    lambda x: 1 if 6 <= int(x) <= 18 else 0
)

df["Traffic_Density"] = df["Vehicle_Count"] / (
    df["Speed"] + 1
)


# =========================
# ENCODING
# =========================
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

df["Weather"] = df["Weather"].map(
    weather_map
)

df["Road_Type"] = df["Road_Type"].map(
    road_map
)

df.dropna(
    inplace=True
)


# =========================
# SCALING
# =========================
scaler = StandardScaler()

scale_columns = [
    "Vehicle_Count",
    "Speed",
    "Traffic_Density"
]

df[scale_columns] = scaler.fit_transform(
    df[scale_columns]
)


# =========================
# SAVE CLEANED DATA
# =========================
df.to_csv(
    "cleaned_traffic_data.csv",
    index=False
)

print("Cleaned dataset saved: cleaned_traffic_data.csv")


# =========================
# TRAIN TEST SPLIT
# =========================
X = df.drop(
    "Accident",
    axis=1
)

y = df["Accident"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# MODEL TRAINING
# =========================
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# =========================
# MODEL EVALUATION
# =========================
predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# =========================
# SAVE MODEL AND SCALER
# =========================
joblib.dump(
    model,
    "traffic_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model saved: traffic_model.pkl")
print("Scaler saved: scaler.pkl")


# =========================
# EDA GRAPHS FROM ACTUAL RAW DATA
# =========================
eda_df = raw_df.copy()

eda_df["Weather"] = eda_df["Weather"].fillna(
    eda_df["Weather"].mode()[0]
)

eda_df["Road_Type"] = eda_df["Road_Type"].fillna(
    eda_df["Road_Type"].mode()[0]
)

eda_df["Speed"] = eda_df["Speed"].fillna(
    eda_df["Speed"].median()
)

eda_df.drop_duplicates(
    inplace=True
)


# Accident Distribution
plt.figure(figsize=(7, 5))
eda_df["Accident"].value_counts().sort_index().plot(
    kind="bar"
)
plt.title("Accident Distribution")
plt.xlabel("Accident Class (0 = No Accident, 1 = Accident)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(
    "accident_distribution.png"
)
plt.close()


# Traffic Time Analysis
plt.figure(figsize=(7, 5))
eda_df.groupby("Time")["Accident"].sum().plot(
    kind="line",
    marker="o"
)
plt.title("Accidents by Time")
plt.xlabel("Hour of Day")
plt.ylabel("Accident Count")
plt.tight_layout()
plt.savefig(
    "traffic_time_analysis.png"
)
plt.close()


# Weather Accident Analysis
plt.figure(figsize=(7, 5))
eda_df.groupby("Weather")["Accident"].sum().plot(
    kind="bar"
)
plt.title("Weather Impact on Accidents")
plt.xlabel("Weather")
plt.ylabel("Accident Count")
plt.tight_layout()
plt.savefig(
    "weather_accident_analysis.png"
)
plt.close()


print("EDA graphs saved:")
print("accident_distribution.png")
print("traffic_time_analysis.png")
print("weather_accident_analysis.png")


# =========================
# FINAL STATUS
# =========================
print()
print("Training Pipeline Completed Successfully.")
print("Now run: python -m streamlit run app.py")