# import pandas as pd
#
# data = pd.read_csv('combined_file_with_all_columns.csv')
# features = data.values
#
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import matplotlib.pyplot as plt
#
# min_samples = 4
#
# # בניית מודל k-nearest neighbors
# neighbors = NearestNeighbors(n_neighbors=min_samples)
# neighbors_fit = neighbors.fit(features)
# distances, indices = neighbors_fit.kneighbors(features)
#
# # מיון המרחקים בסדר יורד
# distances = np.sort(distances[:, -1], axis=0)
#
# # יצירת גרף הברך
# plt.plot(distances)
# plt.title("Elbow Method for Epsilon")
# plt.xlabel("Points")
# plt.ylabel("k-Distance")
# plt.show()
#
# from sklearn.cluster import DBSCAN
#
# # ערכי eps ו-min_samples הנבחרים
# #min_samples = dim + 1
# eps = chosen_eps  # ערך מהגרף
# min_samples = chosen_min_samples  # לפי כלל האצבע או ניסוי
#
# # הרצת DBSCAN
# dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
# clusters = dbscan.fit_predict(features)
#
# # הוספת האשכולות לנתונים
# data['cluster'] = clusters
#
# # הדפסת אשכולות
# print(data.groupby('cluster').size())
#
# from sklearn.metrics import silhouette_score, davies_bouldin_score
#
# # חישוב מדדי איכות
# silhouette = silhouette_score(features, clusters)
# davies_bouldin = davies_bouldin_score(features, clusters)
#
# print(f"Silhouette Score: {silhouette}")
# print(f"Davies-Bouldin Index: {davies_bouldin}")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# Load the combined file
file_path = 'combined_file_with_all_columns.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Preprocess the data: Ensure numerical values and handle missing data
data = data.fillna(0)  # Replace NaN with 0 or use other imputation methods
data_numeric = data.select_dtypes(include=['number'])  # Select only numerical columns
# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)
from sklearn.neighbors import NearestNeighbors

# Determine the k-distance (k = min_samples - 1)
min_samples = 4  # Start with a default value
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(data_scaled)
distances, indices = neighbors_fit.kneighbors(data_scaled)

# Sort distances and plot
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.title("k-Distance Plot")
plt.xlabel("Data Points (sorted)")
plt.ylabel(f"Distance to {min_samples}-th Nearest Neighbor")
plt.show()
#
# # After inspecting the plot, choose the `eps` value manually
# chosen_eps = float(input("Enter chosen eps value based on the plot: "))
# # Run DBSCAN
# dbscan = DBSCAN(eps=chosen_eps, min_samples=min_samples, metric='euclidean')
# clusters = dbscan.fit_predict(data_scaled)
#
# # Add cluster labels to the original data
# data['Cluster'] = clusters
# # Save the clustered data
# output_path = 'dbscan_clustered_data.xlsx'  # Replace with your desired output path
# data.to_excel(output_path, index=False)
#
# print(f"Clustered data saved to {output_path}")


