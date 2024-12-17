import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Function to load and shuffle data
def load_and_shuffle_data(file_path):
    data = pd.read_csv(file_path)
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return shuffled_data


# Function to split features and labels
def split_features_labels(data, label_column='Sheet'):
    features = data.drop(columns=[label_column, 'RowIndex'])
    labels = data[label_column]
    return features, labels


# Function to normalize features
def normalize_features(features):
    return normalize(features)


# Function to encode labels
def encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels)


# Function for UMAP dimensionality reduction
def reduce_dimensions(features):
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2, random_state=42)
    return umap_reducer.fit_transform(features)


# Function to compute cosine similarity matrix
def compute_cosine_similarity(features):
    return cosine_similarity(features)


# Function to fit the Gaussian Mixture Model
def fit_gmm(features, n_components=4):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='spherical',
        init_params='random',
        random_state=42
    )
    return gmm.fit_predict(features)


# Function to calculate evaluation metrics
def calculate_metrics(labels, gmm_labels):
    precision = precision_score(labels, gmm_labels, average='weighted')
    recall = recall_score(labels, gmm_labels, average='weighted')
    f1 = f1_score(labels, gmm_labels, average='weighted')
    accuracy = accuracy_score(labels, gmm_labels)
    return precision, recall, f1, accuracy


# Function to plot UMAP results
def plot_umap(file_path, umap_embedding, labels, gmm_labels):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Set the main title for the entire figure
    plt.suptitle(f"Mixture of Gaussian on {file_path}", fontsize=16)

    # True labels
    plt.subplot(1, 2, 1)
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels, cmap='viridis', s=10)
    plt.title('UMAP with True Labels')
    plt.colorbar()

    # GMM predicted clusters
    plt.subplot(1, 2, 2)
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=gmm_labels, cmap='plasma', s=10)
    plt.title('UMAP with GMM Clusters (Cosine Similarity)')
    plt.colorbar()

    plt.show()


# Main function to run the entire process
def main(file_path):
    # Load and shuffle data
    data = load_and_shuffle_data(file_path)

    # Split into features and labels
    features, labels = split_features_labels(data)

    # Normalize features using cosine similarity
    features_normalized = normalize_features(features)

    #  features_normalized =features

    # Compute cosine similarity matrix
    # cosine_sim_matrix = compute_cosine_similarity(features_normalized)
    #
    # # Print a small sample of the cosine similarity matrix
    # print("Cosine Similarity Matrix (sample):")
    # print(cosine_sim_matrix[:5, :5])

    # Encode labels
    labels_encoded = encode_labels(labels)

    # Dimensionality reduction using UMAP
    umap_embedding = reduce_dimensions(features_normalized)

    # Fit Gaussian Mixture Model (GMM)
    gmm_labels = fit_gmm(features_normalized)  # Use the normalized features, as cosine similarity is already handled in preprocessing

    # Calculate metrics
    precision, recall, f1, accuracy = calculate_metrics(labels_encoded, gmm_labels)

    # Print metrics
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # Plot UMAP results
    plot_umap(file_path, umap_embedding, labels_encoded, gmm_labels)


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
        #main("IR-Newspapers-files\IR-files/word2vec/w2v_lemma_withIDF_withoutStopWords.csv")
        main(full_path)



