import pandas as pd

data = pd.read_csv('combined_file_with_all_columns.csv')
features = data.values

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

min_samples = 4

# בניית מודל k-nearest neighbors
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(features)
distances, indices = neighbors_fit.kneighbors(features)

# מיון המרחקים בסדר יורד
distances = np.sort(distances[:, -1], axis=0)

# יצירת גרף הברך
plt.plot(distances)
plt.title("Elbow Method for Epsilon")
plt.xlabel("Points")
plt.ylabel("k-Distance")
plt.show()

from sklearn.cluster import DBSCAN

# ערכי eps ו-min_samples הנבחרים
#min_samples = dim + 1
eps = chosen_eps  # ערך מהגרף
min_samples = chosen_min_samples  # לפי כלל האצבע או ניסוי

# הרצת DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
clusters = dbscan.fit_predict(features)

# הוספת האשכולות לנתונים
data['cluster'] = clusters

# הדפסת אשכולות
print(data.groupby('cluster').size())

from sklearn.metrics import silhouette_score, davies_bouldin_score

# חישוב מדדי איכות
silhouette = silhouette_score(features, clusters)
davies_bouldin = davies_bouldin_score(features, clusters)

print(f"Silhouette Score: {silhouette}")
print(f"Davies-Bouldin Index: {davies_bouldin}")
