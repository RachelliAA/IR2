import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
#svm
# Load the CSV
input_file = 'your_file.csv'  # Update with the actual path to your file
data = pd.read_csv(input_file)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare features and labels
features = data.drop(columns=['Sheet', 'RowIndex'])  # Drop non-feature columns
labels = data['Sheet']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Build the SVM model with cosine similarity
model = make_pipeline(
    StandardScaler(),  # Normalize the data
    SVC(kernel='linear')  # Linear kernel approximates cosine similarity after scaling
)

# Perform Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=10)
y_pred = cross_val_predict(model, features, labels, cv=cv)

# Calculate metrics
accuracy = accuracy_score(labels, y_pred)
f1 = f1_score(labels, y_pred, average='weighted')
precision = precision_score(labels, y_pred, average='weighted')
recall = recall_score(labels, y_pred, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Save the results (if needed)
results = pd.DataFrame({
    "True Labels": labels,
    "Predicted Labels": y_pred
})
results.to_csv('svm_results.csv', index=False)
