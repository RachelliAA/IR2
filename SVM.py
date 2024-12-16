# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, cross_val_predict
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from sklearn.metrics.pairwise import cosine_similarity
# import umap
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load the CSV
# input_file = 'IR-Newspapers-files/IR-files/bert-sbert/bert_withIDF.csv'
# data = pd.read_csv(input_file)
#
# # Shuffle the data
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Prepare features and labels
# features = data.drop(columns=['Sheet', 'RowIndex'])  # Drop non-feature columns
# labels = data['Sheet']
#
# # Normalize the features
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)
#
# # Compute the cosine similarity matrix
# cosine_sim_matrix = cosine_similarity(features_scaled)
#
# # Encode labels to numerical values
# le = LabelEncoder()
# labels_encoded = le.fit_transform(labels)
#
# # Define the SVM model with precomputed kernel
# model = SVC(kernel='precomputed')
#
# # Perform Stratified K-Fold Cross-Validation
# cv = StratifiedKFold(n_splits=10)
# y_pred = cross_val_predict(model, cosine_sim_matrix, labels_encoded, cv=cv)
#
# # Calculate metrics
# accuracy = accuracy_score(labels_encoded, y_pred)
# f1 = f1_score(labels_encoded, y_pred, average='weighted')
# precision = precision_score(labels_encoded, y_pred, average='weighted')
# recall = recall_score(labels_encoded, y_pred, average='weighted')
#
# # Print metrics
# print(f"Accuracy: {accuracy:.2f}")
# print(f"F1 Score: {f1:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
#
# # UMAP Dimensionality Reduction
# umap_reducer = umap.UMAP(n_components=2, random_state=42)
# features_2d = umap_reducer.fit_transform(features_scaled)
#
# # Create a figure
# plt.figure(figsize=(20, 8))
#
# # Plot UMAP results with true labels
# plt.subplot(1, 2, 1)  # First subplot
# sns.scatterplot(
#     x=features_2d[:, 0], y=features_2d[:, 1],
#     hue=labels_encoded, palette="tab10", s=50, alpha=0.7, edgecolor="k"
# )
# plt.title("True Labels", fontsize=16)
# plt.xlabel("UMAP Dimension 1", fontsize=12)
# plt.ylabel("UMAP Dimension 2", fontsize=12)
# plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
#
# # Plot UMAP results with predicted labels
# plt.subplot(1, 2, 2)  # Second subplot
# sns.scatterplot(
#     x=features_2d[:, 0], y=features_2d[:, 1],
#     hue=y_pred, palette="tab10", s=50, alpha=0.7, edgecolor="k"
# )
# plt.title("Predicted Labels", fontsize=16)
# plt.xlabel("UMAP Dimension 1", fontsize=12)
# plt.ylabel("UMAP Dimension 2", fontsize=12)
# plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
#
# # Adjust layout and ensure colorbar and legend are visible
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
input_file = 'IR-Newspapers-files/IR-files/bert-sbert/bert_withIDF.csv'
data = pd.read_csv(input_file)

# Prepare features and labels
features = data.drop(columns=['Sheet', 'RowIndex'])  # Drop non-feature columns
labels = data['Sheet']

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(features_scaled)

# Contribution of each feature to cosine similarity
contributions = []
for i in range(features_scaled.shape[1]):  # Iterate over features
    feature_matrix = np.outer(features_scaled[:, i], features_scaled[:, i])
    contribution = np.abs(feature_matrix).sum()  # Absolute contribution across all pairs
    contributions.append(contribution)

# Normalize contributions to sum to 1
contributions = np.array(contributions)
normalized_contributions = contributions / contributions.sum()

# Get feature names and their importance
feature_importance = pd.DataFrame({
    "Feature": features.columns,
    "Contribution": normalized_contributions
})

# Sort by contribution
feature_importance_sorted = feature_importance.sort_values(by="Contribution", ascending=False)

# Top 20 features
top_20_features = feature_importance_sorted.head(20)
print("Top 20 features based on cosine similarity contribution:")
print(top_20_features)

# Save to Excel
#top_20_features.to_excel("svm_cosine_important_features.xlsx", index=False)

# Plot top 20 features
plt.figure(figsize=(12, 6))
sns.barplot(data=top_20_features, x="Contribution", y="Feature", palette="viridis")
plt.title("Top 20 Features by Contribution to Cosine Similarity", fontsize=16)
plt.xlabel("Contribution", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()
