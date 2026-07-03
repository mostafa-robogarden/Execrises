import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv('datasets/flavors_of_cacao.csv')

# 2. Count of tuples
num_tuples = len(df)
print("Number of tuples:", num_tuples)

# 3. Count of unique company names
num_unique_companies = df['Company'].nunique()
print("Number of unique companies:", num_unique_companies)

# 4. Count of reviews in 2013
df['ReviewDate'] = pd.to_datetime(df['ReviewDate'], errors='coerce')
num_reviews_2013 = df[df['ReviewDate'].dt.year == 2013].shape[0]
print("Number of reviews in 2013:", num_reviews_2013)

# 5. Count missing values per column
missing_per_column = df.isnull().sum()
print("Missing values per column:\n", missing_per_column)

# 6. Histogram of Ratings
plt.figure()
plt.hist(df['Rating'].dropna(), bins=10)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# 7. Scatter plot: Cocoa Percent vs Rating
# Convert “70%” → 0.70
df['CocoaPercent'] = df['CocoaPercent'].str.rstrip('%').astype(float) / 100
plt.figure()
plt.scatter(df['CocoaPercent'], df['Rating'], alpha=0.6)
plt.title('Cocoa Percent vs Rating')
plt.xlabel('Cocoa Percent')
plt.ylabel('Rating')
plt.show()

# 8. Normalize the ratings column
min_r, max_r = df['Rating'].min(), df['Rating'].max()
df['Normalized Rating'] = (df['Rating'] - min_r) / (max_r - min_r)
print("\nSample of normalized ratings:")
print(df[['Rating', 'Normalized Rating']].head())
