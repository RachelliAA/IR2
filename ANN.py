import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import CosineSimilarity
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

# Step 1: Load and preprocess the data
data = pd.read_csv(file_name)

# Extract input features (all columns except 'Sheet' and 'RowIndex') and labels ('Sheet')
X = data.drop(columns=['Sheet', 'RowIndex']).values
y = data['Sheet'].values

# Encode categorical labels (if needed)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert y to one-hot encoding for CosineSimilarity metric
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(np.unique(y)))

# Number of unique classes
num_classes = len(np.unique(y))
print(f'Number of classes: {num_classes}')

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Step 2: Define a helper function to create models
def create_model(activation_func):
    model = Sequential()
    model.add(Dense(10, activation=activation_func, input_shape=(X_train.shape[1],)))
    model.add(Dense(10, activation=activation_func))
    model.add(Dense(7, activation=activation_func))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', CosineSimilarity()])
    return model

# Define GELU activation
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

# Step 3: Create models
model_relu = create_model('relu')
model_gelu = create_model(gelu)

# Step 4: Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint_relu = ModelCheckpoint('best_model_relu.keras', monitor='val_loss', save_best_only=True)
checkpoint_gelu = ModelCheckpoint('best_model_gelu.keras', monitor='val_loss', save_best_only=True)

# Step 5: Train both models
history_relu = model_relu.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val),
                              callbacks=[early_stop, checkpoint_relu])
history_gelu = model_gelu.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val),
                              callbacks=[early_stop, checkpoint_gelu])

# Step 6: Get predictions for the test set
y_pred_relu = model_relu.predict(X_test)
y_pred_gelu = model_gelu.predict(X_test)

# Step 7: Compare validation accuracies
best_acc_relu = max(history_relu.history['val_accuracy'])
best_acc_gelu = max(history_gelu.history['val_accuracy'])

print(f'Best validation accuracy for ReLU model: {best_acc_relu}')
print(f'Best validation accuracy for GELU model: {best_acc_gelu}')

# Step 8: t-SNE visualization for both models
tsne_relu = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
tsne_gelu = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

tsne_result_relu = tsne_relu.fit_transform(y_pred_relu)
tsne_result_gelu = tsne_gelu.fit_transform(y_pred_gelu)

# Plot t-SNE results
plt.figure(figsize=(8, 4))

# ReLU scatter plot
plt.subplot(1, 2, 1)
plt.scatter(tsne_result_relu[:, 0], tsne_result_relu[:, 1], c=np.argmax(y_test, axis=1), cmap='viridis', s=50)
plt.title('t-SNE Visualization for ReLU Model')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar()

# GELU scatter plot
plt.subplot(1, 2, 2)
plt.scatter(tsne_result_gelu[:, 0], tsne_result_gelu[:, 1], c=np.argmax(y_test, axis=1), cmap='viridis', s=50)
plt.title('t-SNE Visualization for GELU Model')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar()

plt.tight_layout()
plt.show()

# Step 9: Classification report for the better model
if best_acc_relu > best_acc_gelu:
    print("Using ReLU model for evaluation.")
    best_model_predictions = y_pred_relu
else:
    print("Using GELU model for evaluation.")
    best_model_predictions = y_pred_gelu

# Convert predictions to class labels
y_pred_class = np.argmax(best_model_predictions, axis=1)
y_true_class = np.argmax(y_test, axis=1)

report = classification_report(y_true_class, y_pred_class, target_names=label_encoder.classes_)
print("Classification Report for the Best Model Based on Validation Accuracy:\n")
print(report)
