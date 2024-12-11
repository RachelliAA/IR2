import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions
reduced_data = pca.fit_transform(data_numeric.values)

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 7))
for cluster in range(4):  # Adjust for your number of clusters
    cluster_data = reduced_data[data['Cluster'] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}', alpha=0.7)

plt.title('Clusters Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
#
# # Save the clustered data to a new file
# output_path = 'clustered_data_sparse.xlsx'  # Replace with the desired output path
# data.to_excel(output_path, index=False)
#
# print(f"Clustered data saved to {output_path}")
