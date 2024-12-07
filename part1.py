# import pandas as pd
#
# # Upload your Excel file
# file_path = 'tfidf_structured_matrix1.xlsx'  # Replace with the path to your Excel file
#
# # Read all sheets into a dictionary
# sheets = pd.read_excel(file_path, sheet_name=None)
#
# # Get a set of all unique column names from all tabs
# all_columns = set()
# for sheet in sheets.values():
#     all_columns.update(sheet.columns)
#
# # Combine all sheets, filling missing columns with zeros
# combined_df = pd.DataFrame(columns=sorted(all_columns))  # Create a DataFrame with all possible columns
# for sheet_name, df in sheets.items():
#     # Align columns with the combined DataFrame, fill missing with zeros
#     df = df.reindex(columns=combined_df.columns, fill_value=0)
#     combined_df = pd.concat([combined_df, df], ignore_index=True)
#
# # Save the combined sheet to a new Excel file
# output_path = 'combined_file_with_all_columns.xlsx'  # Replace with the desired output path
# combined_df.to_excel(output_path, index=False)
#
# print(f"Combined sheet with all columns saved to {output_path}")
################## combined the tabs to 1 tab #####################################
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

# Load the combined file
file_path = 'combined_file_with_all_columns.xlsx'  # Replace with your combined file path
data = pd.read_excel(file_path)

# Preprocess the data: Ensure numerical values and handle missing data
data = data.fillna(0)  # Replace NaN with 0 or use other imputation methods
data_numeric = data.select_dtypes(include=['number'])  # Select only numerical columns

# Convert to sparse matrix (if necessary, for large datasets)
data_sparse = csr_matrix(data_numeric.values)

# Apply MiniBatchKMeans clustering (this can handle sparse matrices)
kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=100)
data['Cluster'] = kmeans.fit_predict(data_sparse)

# Save the clustered data to a new file
output_path = 'clustered_data_sparse.xlsx'  # Replace with the desired output path
data.to_excel(output_path, index=False)

print(f"Clustered data saved to {output_path}")
############# kmeans clustering worked ##################