import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("data.csv")
df = df.drop(columns=["song_title", "artist"])
X = df.drop(columns=["target"]).values
y = df["target"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump({"X": X_scaled, "y": y, "scaler": scaler, "feature_names": df.drop(columns=["target"]).columns.tolist()}, "spotify_scaled_data.pkl")
joblib.dump(scaler, "scaler.pkl")