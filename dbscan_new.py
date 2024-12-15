import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Read data from a CSV file
def read_csv_data(file_path):
    data = pd.read_csv(file_path)
    # Assuming the CSV file has no header and the first two columns are the features
    return data.iloc[:, 0:2].values, data.iloc[:, 2].values if data.shape[1] > 2 else None

file_path = 'files/bert_withIDF.csv'  # Replace with your CSV file path
X, true_labels = read_csv_data(file_path)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], s=10, color="gray")
plt.title("Data from CSV")
plt.show()

# Perform DBSCAN clustering
esp = 0.1
min_samples = 4

# Fit DBSCAN model
dbscan = DBSCAN(eps=esp, min_samples=min_samples)
labels = dbscan.fit_predict(X)

# Assign noise points (-1) to a separate class for evaluation
evaluated_labels = labels.copy()
evaluated_labels[evaluated_labels == -1] = len(set(labels)) - (1 if -1 in labels else 0)

# Calculate metrics if true labels are available
if true_labels is not None:
    precision = precision_score(true_labels, evaluated_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, evaluated_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, evaluated_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(true_labels, evaluated_labels)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
else:
    print("True labels not provided in the CSV file.")

# Plot DBSCAN results
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
colors = [plt.cm.jet(each) for each in np.linspace(0, 1, len(unique_labels))]

for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points
        color = [0, 0, 0, 1]
    mask = (labels == label)
    plt.scatter(X[mask, 0], X[mask, 1], s=10, color=color, label=f"Cluster {label}" if label != -1 else "Noise")

plt.title(f"DBSCAN Clustering (eps={esp}, min_samples={min_samples})")
plt.legend()
plt.show()
