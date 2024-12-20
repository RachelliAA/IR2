import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler, normalize, LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(input_file):
    data = pd.read_csv(input_file)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    features = data.drop(columns=['Sheet', 'RowIndex'])  # Drop non-feature columns
    labels = data['Sheet']
    return features, labels


def preprocess_features(features):
    # Normalize the features
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled


def apply_pca(features_scaled, n_components=4):
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    return features_pca


def find_optimal_eps(features_normalized):
    # Find optimal eps by analyzing nearest neighbor distances
    neighbors = NearestNeighbors(n_neighbors=5, metric='cosine')
    neighbors.fit(features_normalized)
    distances, _ = neighbors.kneighbors(features_normalized)

    # Sort the distances (for the 4th neighbor)
    k_distances = np.sort(distances[:, 4])

    # Find the Elbow Point for eps
    knee_locator = KneeLocator(range(1, len(k_distances) + 1), k_distances, curve="convex", direction="increasing")
    optimal_eps = k_distances[knee_locator.knee]
    return optimal_eps, k_distances


def apply_dbscan(features_normalized, eps, min_samples=8):
    dbscan = DBSCAN(eps=eps, min_samples=8, metric='cosine')
    dbscan_labels = dbscan.fit_predict(features_normalized)
    return dbscan_labels


def evaluate_clustering(labels_encoded, dbscan_labels):
    accuracy = accuracy_score(labels_encoded, dbscan_labels)
    precision = precision_score(labels_encoded, dbscan_labels, average='weighted', zero_division=0)
    recall = recall_score(labels_encoded, dbscan_labels, average='weighted', zero_division=0)
    f1 = f1_score(labels_encoded, dbscan_labels, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1


def visualize_clusters(features_2d, dbscan_labels, labels_encoded, input_file):

    # Create subplots for visualization (2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot UMAP results with DBSCAN clustering labels
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=dbscan_labels, palette="tab10", s=50, alpha=0.7,
                    edgecolor="k", ax=axes[0])
    axes[0].set_title("DBSCAN Clustering", fontsize=16)
    axes[0].set_xlabel("UMAP Dimension 1", fontsize=12)
    axes[0].set_ylabel("UMAP Dimension 2", fontsize=12)
    axes[0].legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot UMAP results with true labels
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels_encoded, palette="tab10", s=50, alpha=0.7,
                    edgecolor="k", ax=axes[1])
    axes[1].set_title("True Labels", fontsize=16)
    axes[1].set_xlabel("UMAP Dimension 1", fontsize=12)
    axes[1].set_ylabel("UMAP Dimension 2", fontsize=12)
    axes[1].legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and ensure colorbar and legend are visible
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f'running DBSCAN on {input_file}', fontsize=20)
    #plt.show()


def main(input_file):
    # Load the data
    features, labels = load_data(input_file)

    # Preprocess the features
    features_scaled = preprocess_features(features)

    # Apply PCA for dimensionality reduction (optional)
    #features_scaled = apply_pca(features_scaled, n_components=4)

    # Encode labels to numeric values
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Normalize the features
    features_normalized = normalize(features_scaled)

    # UMAP Dimensionality Reduction
    umap_reducer = umap.UMAP(n_neighbors=10, metric='cosine', min_dist=0.2, n_components=4, random_state=42)
    features_2d = umap_reducer.fit_transform(features_normalized)


    # Find the optimal eps value
    optimal_eps, k_distances = find_optimal_eps(features_2d)
    print(f"epsilon = {optimal_eps}")
    # Plot the k-distance graph
    plt.figure(figsize=(8, 6))
    plt.plot(np.sort(k_distances), marker='o')
    plt.title('K-distance Graph for DBSCAN', fontsize=16)
    plt.suptitle(f'epsilon= {optimal_eps}', fontsize=14)
    plt.xlabel('Data Points', fontsize=12)
    plt.ylabel('4th Nearest Neighbor Distance', fontsize=12)
    plt.grid(True)
    #plt.show()

    # Apply DBSCAN clustering with tuned parameters
    dbscan_labels = apply_dbscan(features_2d, eps=optimal_eps, min_samples=4)

    # Evaluate the clustering performance
    accuracy, precision, recall, f1 = evaluate_clustering(labels_encoded, dbscan_labels)

    # Print metrics
    print(f"Running DBSCAN on {input_file}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Visualize the clustering results
    visualize_clusters(features_2d, dbscan_labels, labels_encoded, input_file)

    # Print out the number of clusters found by DBSCAN
    print(f"Number of clusters found: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")


import os

# Define the root folder
root_folder = "IR-Newspapers-files"

# List of files with their relative paths
files = [
    "IR-files\\bert-sbert\\bert_withIDF.csv",
    "IR-files\\bert-sbert\\bert_withoutIDF.csv",
    "IR-files\\bert-sbert\\sbert_vectors.csv",
    "IR-files\\doc2vec\\doc2vec_vectors.csv",
    "IR-files\\glove\\glove_clean_withIDF_withoutStopWords.csv",
    "IR-files\\glove\\glove_clean_withIDF_withStopWords.csv",
    "IR-files\\glove\\glove_clean_withoutIdf_withoutStopWords.csv",
    "IR-files\\glove\\glove_clean_withoutIdf_withStopWords.csv",
    "IR-files\\glove\\glove_lemma_withIDF_withoutStopWords.csv",
    "IR-files\\glove\\glove_lemma_withIDF_withStopWords.csv",
    "IR-files\\glove\\glove_lemma_withoutIdf_withoutStopWords.csv",
    "IR-files\\glove\\glove_lemma_withoutIdf_withStopWords.csv",
    "IR-files\\word2vec\\w2v_clean_withIDF_withoutStopWords.csv",
    "IR-files\\word2vec\\w2v_clean_withIDF_withStopWords.csv",
    "IR-files\\word2vec\\w2v_clean_withoutIdf_withoutStopWords.csv",
    "IR-files\\word2vec\\w2v_clean_withoutIdf_withStopWords.csv",
    "IR-files\\word2vec\\w2v_lemma_withIDF_withoutStopWords.csv",
    "IR-files\\word2vec\\w2v_lemma_withIDF_withStopWords.csv",
    "IR-files\\word2vec\\w2v_lemma_withoutIdf_withoutStopWords.csv",
    "IR-files\\word2vec\\w2v_lemma_withoutIdf_withStopWords.csv",
]

if __name__ == "__main__":
    for file_path in files:
        # Construct the full file path with the root folder
        full_path = os.path.join(root_folder, file_path.replace("\\", "/"))

        print(f"Processing file: {full_path}")
        main(full_path)
    plt.show()

