import pandas as pd
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



def main1(input_file):
    print("second")
    # Load CSV
    data = pd.read_csv(input_file)

    # Shuffle the data
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split features and labels
    features = shuffled_data.drop(columns=['Sheet', 'RowIndex'])
    labels = shuffled_data['Sheet']


    # #Preprocess data
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
    umap_reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, n_components=5, random_state=42)
    umap_embedding = umap_reducer.fit_transform(features_normalized)

    # Fit KMeans Model
    kmeans = KMeans(n_clusters=4,init = "random",max_iter=500, random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_embedding)


    # # Transform to cosine similarity matrix
    # cosine_sim_matrix = cosine_similarity(features_normalized)
    #
    # # Run KMeans using cosine similarity indirectly (on similarity matrix)
    # kmeans = KMeans(n_clusters=n_classes, random_state=42)
    # kmeans_labels = kmeans.fit_predict(cosine_sim_matrix)


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

    #Plot UMAP with clusters
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

    #plt.show()


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
    plt.show()


