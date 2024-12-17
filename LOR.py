import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def lor(path):
    # Load the data
    data = pd.read_csv(path)  # If data is in a CSV file

    df = pd.DataFrame(data)

    # Encode the labels into numerical values
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Sheet'])  # Label encode the target column

    # Separate features (all Dim columns) and target variable
    X = df.drop(columns=['Sheet', 'Label'])  # Drop the label and target column
    y = df['Label'].values  # Use the multi-class labels for classification

    # Step 4: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Compute cosine similarity
    cosine_sim_matrix = cosine_similarity(X_scaled)

    # Step 6: Split the cosine similarity matrix into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(cosine_sim_matrix, y, test_size=0.2, random_state=42,
                                                        stratify=y)

    # Step 7: Train Logistic Regression model with 10-fold cross-validation
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

    # 10-fold Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

    # Output cross-validation results
    print("10-fold Cross Validation Accuracy Scores:", cv_scores)
    print("Mean Accuracy from Cross Validation:", np.mean(cv_scores))

    # Step 8: Train the model on the entire training set
    model.fit(X_train, y_train)

    # Step 9: Make predictions
    y_pred = model.predict(X_test)

    # Step 10: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=le.classes_)

    print("Accuracy on Test Set:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_rep)

    # Step 11: Visualize the results using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(cosine_sim_matrix)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8, s=50)
    plt.colorbar(scatter, label='Classes')
    plt.title('t-SNE Visualization of Cosine Similarity Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

    model.fit(X_scaled, y)

    # Get the absolute value of the coefficients
    coefficients = np.abs(model.coef_[0])  # model.coef_ gives the coefficients for each class, use the first class

    # Get the indices of the top 20 features with the highest absolute coefficient values
    top_20_indices = np.argsort(coefficients)[::-1][:20]

    # Get the feature names from the original X dataset
    top_20_features = X.columns[top_20_indices]

    print("Top 20 features and their significance:")
    for feature, coef in zip(top_20_features, coefficients[top_20_indices]):
        print(f"{feature}: {coef}")


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
    lor(full_path)
