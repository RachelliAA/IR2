import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
################## GOOD CODE FOR GUASSON DISTRABUTION #########################
# # Load CSV
# #data = pd.read_csv('IR-Newspapers-files/IR-files/bert-sbert/bert_withIDF.csv')
# data = pd.read_csv('IR-Newspapers-files/IR-files/bert-sbert/bert_withoutIDF.csv')
#
# #Shuffle the data
# shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Split features and labels
# features = shuffled_data.drop(columns=['Sheet', 'RowIndex'])
# labels = shuffled_data['Sheet']
#
#
#
# # # Preprocess data
# # features = data.iloc[:, 2:]  # Select columns Dim0 to DimN
# # labels = data['Sheet']  # Select the Sheet column
#
# # Normalize features
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)
#
# # Encode labels to numerical values
# le = LabelEncoder()
# labels_encoded = le.fit_transform(labels)
#
# # Dimensionality reduction using UMAP
# umap_reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
# umap_embedding = umap_reducer.fit_transform(features_scaled)
#
# # Fit Gaussian Mixture Model
# n_classes = len(np.unique(labels_encoded))
# #print(n_classes)
#
# gmm = GaussianMixture(n_components=n_classes, random_state=42)
# #gmm = GaussianMixture(n_components=n_classes, random_state=42)
# gmm_labels = gmm.fit_predict(features_scaled)
#
# # Calculate metrics
# precision = precision_score(labels_encoded, gmm_labels, average='weighted')
# recall = recall_score(labels_encoded, gmm_labels, average='weighted')
# f1 = f1_score(labels_encoded, gmm_labels, average='weighted')
# accuracy = accuracy_score(labels_encoded, gmm_labels)
#
# # Print metrics
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")
# print(f"Accuracy: {accuracy:.2f}")
#
# # Plot UMAP with clusters
# plt.figure(figsize=(12, 6))
# sns.set(style="whitegrid")
#
# # True labels
# plt.subplot(1, 2, 1)
# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels_encoded, cmap='viridis', s=10)
# plt.title('UMAP with True Labels')
# plt.colorbar()
#
# # GMM predicted clusters
# plt.subplot(1, 2, 2)
# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=gmm_labels, cmap='plasma', s=10)
# plt.title('UMAP with GMM Clusters')
# plt.colorbar()
#
# plt.show()


#################3 good code with cosine similarity ##################
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
data = pd.read_csv('IR-Newspapers-files/IR-files/bert-sbert/bert_withoutIDF.csv')

# Shuffle the data
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and labels
features = shuffled_data.drop(columns=['Sheet', 'RowIndex'])
labels = shuffled_data['Sheet']

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Normalize features to use cosine similarity
features_normalized = normalize(features_scaled)

# Encode labels to numerical values
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Dimensionality reduction using UMAP
umap_reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
umap_embedding = umap_reducer.fit_transform(features_normalized)

# Fit Gaussian Mixture Model
n_classes = len(np.unique(labels_encoded))
gmm = GaussianMixture(n_components=n_classes, random_state=42)
gmm_labels = gmm.fit_predict(features_normalized)

# Calculate metrics
precision = precision_score(labels_encoded, gmm_labels, average='weighted')
recall = recall_score(labels_encoded, gmm_labels, average='weighted')
f1 = f1_score(labels_encoded, gmm_labels, average='weighted')
accuracy = accuracy_score(labels_encoded, gmm_labels)

# Print metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Plot UMAP with clusters
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

# True labels
plt.subplot(1, 2, 1)
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels_encoded, cmap='viridis', s=10)
plt.title('UMAP with True Labels')
plt.colorbar()

# GMM predicted clusters
plt.subplot(1, 2, 2)
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=gmm_labels, cmap='plasma', s=10)
plt.title('UMAP with GMM Clusters (Cosine Similarity)')
plt.colorbar()

plt.show()
