
import pandas as pd

# Load Heart Disease dataset
df = pd.read_csv("Heart.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)
# Display all column names
print("\nColumns in the dataset:")
print(df.columns.tolist())

sklearn.preprocessing
