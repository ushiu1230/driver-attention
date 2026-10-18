import pandas as pd

csv_path = "path/to/sim_gaze_df.csv"
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))

print("\nFirst 5 rows:")
print(df.head())

# Kiểm tra cột Frame nếu có
if "Frame" in df.columns:
    frame = pd.to_numeric(df["Frame"], errors="coerce")

    print("\nFrame range:", frame.min(), "to", frame.max())
    print("Missing Frame:", frame.isna().sum())
    print("Duplicate Frame rows:", frame.duplicated(keep=False).sum())