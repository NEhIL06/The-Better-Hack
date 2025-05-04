import pandas as pd

# Read the Excel file
df = pd.read_excel('desifood.xlsx')  # Replace with your actual file name

# Get the first column name
first_col = df.columns[0]

# Create a new column that combines all columns including the first one
df['combined'] = df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Create a new DataFrame with just the first column and the combined column
result_df = df[[first_col, 'combined']]

# Save the result to a new Excel file
result_df.to_excel('merged_output.xlsx', index=False)

print("Done pencho")