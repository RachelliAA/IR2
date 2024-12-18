import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os


def rf(path):

    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Sheet'])

    # Separate features (all Dim columns) and target variable
    X = df.drop(columns=['Sheet', 'Label'])  # Drop the label and target column
    y = df['Label']  # Target (label)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(X_scaled)

    # Convert cosine similarity matrix to a DataFrame
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix)

    # Step 3: Split Data into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(cosine_sim_df, y, test_size=0.3, random_state=42)

    # Step 4: Initialize the Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=100,  # Number of trees in the forest
        max_depth=None,  # Maximum depth of the tree
        random_state=42  # Random state for reproducibility
    )

    # Step 5: Perform 10-Fold Cross-Validation
    cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=10)
    print("10-Fold Cross-Validation Scores:")
    print(cv_scores)
    print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")

    # Step 6: Train the Model on the Full Training Set
    cv_scores.fit(X_train, y_train)

    # Step 7: Make Predictions
    y_pred = rf_clf.predict(X_test)

    # Step 8: Evaluate the Model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Optional: Feature Importance (Using cosine similarity, the features represent rows of similarity scores)
    feature_importances = rf_clf.feature_importances_
    important_features = pd.DataFrame({
        'Feature': [f'Row {i}' for i in range(cosine_sim_df.shape[1])],
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Identify the top 4 most significant attributes
    top_20_features = important_features.head(20)
    print("\nTop 20 Most Significant Features:")
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
    rf(full_path)
