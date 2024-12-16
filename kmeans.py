# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# import umap
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load CSV
# input_file ='IR-Newspapers-files/IR-files/bert-sbert/bert_withIDF.csv'
# # data = pd.read_csv('IR-Newspapers-files/IR-files/bert-sbert/bert_withIDF.csv')
# data = pd.read_csv(input_file)
#
# # Shuffle the data
# shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Split features and labels
# features = shuffled_data.drop(columns=['Sheet', 'RowIndex'])
# labels = shuffled_data['Sheet']
#
# # Preprocess data
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
# # Fit KMeans Model
# n_classes = len(np.unique(labels_encoded))
# kmeans = KMeans(n_clusters=n_classes, random_state=42)
# kmeans_labels = kmeans.fit_predict(features_scaled)
#
# # Calculate metrics
# precision = precision_score(labels_encoded, kmeans_labels, average='weighted')
# recall = recall_score(labels_encoded, kmeans_labels, average='weighted')
# f1 = f1_score(labels_encoded, kmeans_labels, average='weighted')
# accuracy = accuracy_score(labels_encoded, kmeans_labels)
#
# # Print metrics
#
# print(f"Input file {input_file}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")
# print(f"Accuracy: {accuracy:.2f}")
#
# # Plot UMAP with clusters
# plt.figure(figsize=(12, 6))
# sns.set(style="whitegrid")
#
# # Set the main title for the entire figure
# plt.suptitle(f'{input_file}', fontsize=16)
#
# # True labels
# plt.subplot(1, 2, 1)
# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels_encoded, cmap='viridis', s=10)
# plt.title('UMAP with True Labels')
# plt.colorbar()
#
# # KMeans predicted clusters
# plt.subplot(1, 2, 2)
# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=kmeans_labels, cmap='plasma', s=10)
# plt.title('UMAP with KMeans Clusters')
# plt.colorbar()
#
# plt.show()

# ######### good code with cosine similarity ############
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import KMeans
# import numpy as np
# import pandas as pd
# import umap
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#
# # Load CSV
# input_file = 'IR-Newspapers-files/IR-files/bert-sbert/bert_withIDF.csv'
# data = pd.read_csv(input_file)
#
# # Shuffle the data
# shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Split features and labels
# features = shuffled_data.drop(columns=['Sheet', 'RowIndex'])
# labels = shuffled_data['Sheet']
#
# # Preprocess data
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
# # Calculate cosine similarity matrix
# cosine_sim_matrix = cosine_similarity(features_scaled)
#
# # Convert cosine similarity to cosine distance
# cosine_dist_matrix = 1 - cosine_sim_matrix
#
# # Fit KMeans Model using cosine distance
# n_classes = len(np.unique(labels_encoded))
# kmeans = KMeans(n_clusters=n_classes,  random_state=42, max_iter=500)
# kmeans_labels = kmeans.fit_predict(cosine_dist_matrix)
#
# # Calculate metrics
# precision = precision_score(labels_encoded, kmeans_labels, average='weighted')
# recall = recall_score(labels_encoded, kmeans_labels, average='weighted')
# f1 = f1_score(labels_encoded, kmeans_labels, average='weighted')
# accuracy = accuracy_score(labels_encoded, kmeans_labels)
#
# # Print metrics
# print(f"Input file {input_file}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")
# print(f"Accuracy: {accuracy:.2f}")
#
# # Plot UMAP with clusters
# plt.figure(figsize=(12, 6))
# sns.set(style="whitegrid")
#
# # Set the main title for the entire figure
# plt.suptitle(f'{input_file}', fontsize=16)
#
# # True labels
# plt.subplot(1, 2, 1)
# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels_encoded, cmap='viridis', s=10)
# plt.title('UMAP with True Labels')
# plt.colorbar()
#
# # KMeans predicted clusters
# plt.subplot(1, 2, 2)
# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=kmeans_labels, cmap='plasma', s=10)
# plt.title('UMAP with KMeans Clusters (Cosine Distance)')
# plt.colorbar()
#
# plt.show()


###### better cosine kmeans#######################
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load CSV
#input_file = 'IR-Newspapers-files/IR-files/bert-sbert/bert_withIDF.csv'
input_file = 'IR-Newspapers-files/IR-files/word2vec/w2v_clean_withIDF_withoutStopWords.csv'
data = pd.read_csv(input_file)

# Shuffle the data
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and labels
features = shuffled_data.drop(columns=['Sheet', 'RowIndex'])
labels = shuffled_data['Sheet']


#Preprocess data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# # Preprocess data
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# Normalize features to use cosine similarity
features_normalized = normalize(features_scaled)

# Encode labels to numerical values
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Dimensionality reduction using UMAP
umap_reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
umap_embedding = umap_reducer.fit_transform(features_normalized)

# Fit KMeans Model
n_classes = len(np.unique(labels_encoded))
print(f"n= {n_classes}")
# kmeans = KMeans(n_clusters=n_classes, random_state=42)
# kmeans_labels = kmeans.fit_predict(features_normalized)


# Transform to cosine similarity matrix
cosine_sim_matrix = cosine_similarity(features_normalized)

# Run KMeans using cosine similarity indirectly (on similarity matrix)
kmeans = KMeans(n_clusters=n_classes, random_state=42)
kmeans_labels = kmeans.fit_predict(cosine_sim_matrix)


# Calculate metrics
precision = precision_score(labels_encoded, kmeans_labels, average='weighted')
recall = recall_score(labels_encoded, kmeans_labels, average='weighted')
f1 = f1_score(labels_encoded, kmeans_labels, average='weighted')
accuracy = accuracy_score(labels_encoded, kmeans_labels)

# Print metrics
print(f"Kmeans results on {input_file}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Plot UMAP with clusters
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

# Set the main title for the entire figure
plt.suptitle(f'Kmeans on {input_file}', fontsize=16)

# True labels
plt.subplot(1, 2, 1)
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels_encoded, cmap='viridis', s=10)
plt.title('UMAP with True Labels')
plt.colorbar()

# KMeans predicted clusters
plt.subplot(1, 2, 2)
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=kmeans_labels, cmap='plasma', s=10)
plt.title('UMAP with KMeans Clusters')
plt.colorbar()

plt.show()


#accuray 0.30
# # Split features and labels
# features = shuffled_data.drop(columns=['Sheet', 'RowIndex'])
# labels = shuffled_data['Sheet']
# features = features.apply(np.log1p)  # Apply log transformation
#
# # # Preprocess data
# # scaler = StandardScaler()
# # features_scaled = scaler.fit_transform(features)
# #
# # # Normalize features to use cosine similarity
# # features_normalized = normalize(features_scaled)
# features_normalized=features
# # Encode labels to numerical values
# le = LabelEncoder()
# labels_encoded = le.fit_transform(labels)
