import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import os


def nb(path):
    # Load data
    data = pd.read_csv(path)  # If data is in a CSV file
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Sheet'])

    # Separate features and target variable
    X = df.drop(columns=['Sheet', 'Label', 'RowIndex'])  # Features
    y = df['Label']  # Target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(X_scaled)

    # Use cosine similarity matrix as features
    X_cosine_features = pd.DataFrame(cosine_sim_matrix)

    # 10-fold cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Create Naive Bayes classifier
    nb = GaussianNB()

    # Lists to store metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Cross-validation loop
    for train_index, test_index in kf.split(X_cosine_features, y):
        # Split data into training and testing sets
        X_train, X_test = X_cosine_features.iloc[train_index], X_cosine_features.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train Naive Bayes classifier
        nb.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = nb.predict(X_test)

        # Calculate metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Print the average metrics from the 10-fold cross-validation
    print(f'Average Accuracy: {np.mean(accuracies):.4f}')
    print(f'Average Precision: {np.mean(precisions):.4f}')
    print(f'Average Recall: {np.mean(recalls):.4f}')
    print(f'Average F1 Score: {np.mean(f1_scores):.4f}')

    # Reduce features to 2D for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cosine_features)

    # Split reduced data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    nb.fit(X_train, y_train)

    # Create mesh grid for decision boundary
    x_min, x_max = X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1
    y_min, y_max = X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict probabilities for the mesh grid points
    Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', edgecolor='k', s=50)
    plt.title("Naive Bayes Decision Boundary (PCA-Reduced Features)")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.show()

    # Calculate mutual information between features and target variable
    mi_scores = mutual_info_classif(X_scaled, y)

    # Create a DataFrame with feature names and their corresponding MI scores
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Information Score': mi_scores
    })

    # Sort the features by mutual information scores in descending order
    mi_df = mi_df.sort_values(by='Mutual Information Score', ascending=False)

    # Get the top 20 most important features based on mutual information scores
    top_20_features = mi_df.head(20)

    # Print the top 20 most important features
    print("Top 20 Most Important Features Based on Mutual Information:")
    print(top_20_features)


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

for file_path in files:
    # Construct the full file path with the root folder
    full_path = os.path.join(root_folder, file_path.replace("\\", "/"))
    print(f"Processing file: {full_path}")
    nb(full_path)
