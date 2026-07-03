import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/results.csv")

missing_before = df.isna().sum().sum()
df = df.dropna()
missing_after = df.isna().sum().sum()

print(f"Missing values before drop: {missing_before}")
print(f"Missing values after drop:  {missing_after}")

num_rows = len(df)
num_tournaments = df["tournament"].nunique()

print(f"Number of tuples (rows): {num_rows}")
print(f"Number of tournaments:   {num_tournaments}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
invalid_dates = df["date"].isna().sum()
if invalid_dates > 0:
    df = df.dropna(subset=["date"])

matches_2018 = (df["date"].dt.year == 2018).sum()
print(f"Matches played in 2018:  {matches_2018}")

home_wins  = (df["home_score"] > df["away_score"]).sum()
home_losses = (df["home_score"] < df["away_score"]).sum()
draws       = (df["home_score"] == df["away_score"]).sum()

print("\nHome outcomes:")
print(f"  Home wins:  {home_wins}")
print(f"  Home losses:{home_losses}")
print(f"  Draws:      {draws}")

outcomes = pd.Series(
    {"Home Win": home_wins, "Home Loss": home_losses, "Draw": draws}
)
plt.figure()
outcomes.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Home Outcomes (Win/Loss/Draw)")
plt.ylabel("")  # tidy up
plt.tight_layout()
plt.show()

if df["neutral"].dtype == object:
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        ).fillna(df["neutral"])
    )

df["neutral"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, title="Neutral Venue (True/False)"); plt.ylabel(""); plt.tight_layout(); plt.show()
