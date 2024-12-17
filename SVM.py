import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(input_file):
    """
    Loads and preprocesses the data from a CSV file.

    Parameters:
    - input_file (str): Path to the CSV file.

    Returns:
    - features_scaled (np.array): Normalized features.
    - labels_encoded (np.array): Encoded labels.
    - features (pd.DataFrame): Original feature names for further use.
    """
    data = pd.read_csv(input_file).sample(frac=1, random_state=42).reset_index(drop=True)
    features = data.drop(columns=['Sheet', 'RowIndex'])
    labels = data['Sheet']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    return features_scaled, labels_encoded, features


def compute_cosine_similarity(features_scaled):
    """Computes the cosine similarity matrix."""
    return cosine_similarity(features_scaled)


def train_svm_with_cosine_similarity(cosine_sim_matrix, labels_encoded, n_splits=10):
    """
    Trains an SVM using a cosine similarity kernel and performs cross-validation.

    Parameters:
    - cosine_sim_matrix (np.array): Cosine similarity matrix.
    - labels_encoded (np.array): Encoded labels.
    - n_splits (int): Number of splits for cross-validation.

    Returns:
    - y_pred (np.array): Predicted labels from cross-validation.
    - model (SVC): Trained SVM model.
    """
    model = SVC(kernel='precomputed')
    cv = StratifiedKFold(n_splits=n_splits)
    y_pred = cross_val_predict(model, cosine_sim_matrix, labels_encoded, cv=cv)
    model.fit(cosine_sim_matrix, labels_encoded)
    return y_pred, model


def calculate_metrics(labels_encoded, y_pred):
    """Calculates and prints performance metrics."""
    accuracy = accuracy_score(labels_encoded, y_pred)
    f1 = f1_score(labels_encoded, y_pred, average='weighted')
    precision = precision_score(labels_encoded, y_pred, average='weighted')
    recall = recall_score(labels_encoded, y_pred, average='weighted')



    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return accuracy, f1, precision, recall


def calculate_feature_importance(model, features, cosine_sim_matrix, labels_encoded, top_n=20):
    """
    Calculates permutation feature importance.

    Parameters:
    - model (SVC): Trained SVM model.
    - features (pd.DataFrame): Original features.
    - cosine_sim_matrix (np.array): Cosine similarity matrix.
    - labels_encoded (np.array): Encoded labels.
    - top_n (int): Number of top features to return.

    Returns:
    - top_features_df (pd.DataFrame): DataFrame of top N features.
    """
    result = permutation_importance(model, cosine_sim_matrix, labels_encoded, n_repeats=10, random_state=42, n_jobs=-1)
    feature_importances = result.importances_mean[:features.shape[1]]

    features_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})
    top_features_df = features_df.sort_values(by='Importance', ascending=False).head(top_n)

    return top_features_df


def perform_umap(features_scaled):
    """Performs UMAP dimensionality reduction."""
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    return umap_reducer.fit_transform(features_scaled)


def plot_umap(features_2d, labels_encoded, y_pred, input_file):
    """Plots UMAP results with true and predicted labels."""
    plt.figure(figsize=(16, 8))
    sns.set(style="whitegrid")

    plt.suptitle(f"SVM on {input_file}")
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1],
                    hue=labels_encoded, palette="tab10", s=50, alpha=0.7, edgecolor="k")
    plt.title("True Labels")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1],
                    hue=y_pred, palette="tab10", s=50, alpha=0.7, edgecolor="k")
    plt.title("Predicted Labels")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    plt.tight_layout()
    plt.show()


def main(input_file):
    """Main function to run the full pipeline."""
    print(f"SVM on {input_file}")
    features_scaled, labels_encoded, features = load_and_preprocess_data(input_file)
    cosine_sim_matrix = compute_cosine_similarity(features_scaled)
    y_pred, model = train_svm_with_cosine_similarity(cosine_sim_matrix, labels_encoded)
    calculate_metrics(labels_encoded, y_pred)
    top_features_df = calculate_feature_importance(model, features, cosine_sim_matrix, labels_encoded)
    print("\nTop Features:")
    print(top_features_df)

    features_2d = perform_umap(features_scaled)
    plot_umap(features_2d, labels_encoded, y_pred,input_file)


import os

# Define the root folder
root_folder = "IR-Newspapers-files"

# List of files with their relative paths
files = [
    #"IR-files\\bert-sbert\\bert_withIDF.csv",
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

