# 1. Count the zeros specifically
import pandas as pd
df_combined = pd.read_csv(r"C:\Users\shre0\vs_code sign\Project_Sleep_Disorder\Dataset\final_dataset_AP03.csv")  # Load your combined DataFrame
zeros_count = (df_combined['SpO2_Interpolated'] == 0).sum()

# 2. Count the NaNs
nan_count = df_combined['SpO2_Interpolated'].isna().sum()

# 3. Calculate the total rows and percentages
total_rows = len(df_combined)
nan_percent = (nan_count / total_rows) * 100

print(f"--- SpO2 Interpolation Quality Check ---")
print(f"Total Rows: {total_rows}")
print(f"Zeros found: {zeros_count} (Should be 0 if cleaning worked)")
print(f"NaNs found: {nan_count} ({nan_percent:.2f}% of data)")

# 4. Check the range to ensure legitimate values
if not df_combined['SpO2_Interpolated'].dropna().empty:
    min_val = df_combined['SpO2_Interpolated'].min()
    max_val = df_combined['SpO2_Interpolated'].max()
    print(f"Data Range: {min_val}% to {max_val}%")
else:
    print("Warning: The column is entirely empty!")