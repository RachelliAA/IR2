#with umap visualization precision recall and f1
#not run yet!!!!
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix
import umap
import matplotlib.pyplot as plt

# Load the combined file
file_path = 'combined_file_with_all_columns.xlsx'  # Replace with your combined file path
data = pd.read_excel(file_path)

# Assuming 'TrueLabel' column exists in your dataset for evaluation
true_labels = data['TrueLabel'] if 'TrueLabel' in data.columns else None

# Preprocess the data: Ensure numerical values and handle missing data
data = data.fillna(0)  # Replace NaN with 0 or use other imputation methods
data_numeric = data.select_dtypes(include=['number'])  # Select only numerical columns

# Convert to sparse matrix (if necessary, for large datasets)
data_sparse = csr_matrix(data_numeric.values)

# Apply MiniBatchKMeans clustering (this can handle sparse matrices)
kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=100)
data['Cluster'] = kmeans.fit_predict(data_sparse)

# If true labels are available, evaluate performance
if true_labels is not None:
    precision = precision_score(true_labels, data['Cluster'], average='macro')
    recall = recall_score(true_labels, data['Cluster'], average='macro')
    f1 = f1_score(true_labels, data['Cluster'], average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    print("True labels not found. Skipping evaluation.")

# Dimensionality reduction using UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
reduced_data = reducer.fit_transform(data_numeric.values)

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 7))
for cluster in range(4):  # Adjust for your number of clusters
    cluster_data = reduced_data[data['Cluster'] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}', alpha=0.7)

plt.title('Clusters Visualization using UMAP')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Save the clustered data to a new file
output_path = 'clustered_data_umap.xlsx'  # Replace with the desired output path
data.to_excel(output_path, index=False)

print(f"Clustered data saved to {output_path}")
