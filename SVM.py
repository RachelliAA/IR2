import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
input_file = 'IR-Newspapers-files/IR-files/bert-sbert/bert_withIDF.csv'
data = pd.read_csv(input_file)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare features and labels
features = data.drop(columns=['Sheet', 'RowIndex'])  # Drop non-feature columns
labels = data['Sheet']

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Compute the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(features_scaled)

# Encode labels to numerical values
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Define the SVM model with precomputed kernel
model = SVC(kernel='precomputed')

# Perform Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=10)
y_pred = cross_val_predict(model, cosine_sim_matrix, labels_encoded, cv=cv)

# Calculate metrics
accuracy = accuracy_score(labels_encoded, y_pred)
f1 = f1_score(labels_encoded, y_pred, average='weighted')
precision = precision_score(labels_encoded, y_pred, average='weighted')
recall = recall_score(labels_encoded, y_pred, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# UMAP Dimensionality Reduction
umap_reducer = umap.UMAP(n_components=2, random_state=42)
features_2d = umap_reducer.fit_transform(features_scaled)

# Create a figure
plt.figure(figsize=(20, 8))

# Plot UMAP results with true labels
plt.subplot(1, 2, 1)  # First subplot
sns.scatterplot(
    x=features_2d[:, 0], y=features_2d[:, 1],
    hue=labels_encoded, palette="tab10", s=50, alpha=0.7, edgecolor="k"
)
plt.title("True Labels", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=12)
plt.ylabel("UMAP Dimension 2", fontsize=12)
plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot UMAP results with predicted labels
plt.subplot(1, 2, 2)  # Second subplot
sns.scatterplot(
    x=features_2d[:, 0], y=features_2d[:, 1],
    hue=y_pred, palette="tab10", s=50, alpha=0.7, edgecolor="k"
)
plt.title("Predicted Labels", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=12)
plt.ylabel("UMAP Dimension 2", fontsize=12)
plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and ensure colorbar and legend are visible
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
