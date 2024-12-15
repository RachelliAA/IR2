
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the combined file
file_path = 'files/bert_withIDF.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Preprocess the data: Ensure numerical values and handle missing data
data = data.fillna(0)  # Replace NaN with 0 or use other imputation methods
data_numeric = data.select_dtypes(include=['number'])  # Select only numerical columns
# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

min_samples = 4


chosen_eps = 0.1
# Run DBSCAN
dbscan = DBSCAN(eps=chosen_eps, min_samples=min_samples, metric='euclidean')
clusters = dbscan.fit_predict(data_scaled)

# Add cluster labels to the original data
data['Cluster'] = clusters
# Save the clustered data
output_path = 'dbscan_clustered_data.xlsx'  # Replace with your desired output path
data.to_excel(output_path, index=False)

print(f"Clustered data saved to {output_path}")


