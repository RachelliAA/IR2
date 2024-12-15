import tensorflow as tf

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --- Step 1: Read and preprocess the data ---
# File path
file_path = 'IR-files/bert-sbert/bert_withIDF.csv'  # Replace with the actual file path
data = pd.read_csv(file_path, delimiter=',')  # Adjust delimiter if needed

# Extract features and labels
X = data[['Dim0', 'Dim1', 'Dim2', 'Dim3', 'Dim4']].values  # Features
y = data['RowIndex'].values  # Labels (or replace with the correct column)

# Preprocess labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode labels as integers
num_classes = len(label_encoder.classes_)
y = to_categorical(y, num_classes=num_classes)  # One-hot encode labels

# Split data into train/validation/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# --- Step 2: Define the model creation function ---
def create_model(activation):
    model = Sequential([
        Dense(50, activation=activation, input_shape=(X.shape[1],)),  # Input layer
        Dense(10, activation=activation),
        Dense(10, activation=activation),
        Dense(7, activation=activation),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Step 3: Define callbacks ---
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# --- Step 4: Train and evaluate the model with ReLU activation ---
model_relu = create_model('relu')
print("Training model with ReLU activation...")
history_relu = model_relu.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)
accuracy_relu = model_relu.evaluate(X_test, y_test, verbose=0)[1]
print(f"Test accuracy with ReLU activation: {accuracy_relu:.2f}")

# --- Step 5: Train and evaluate the model with GELU activation ---
# GELU isn't natively supported in TensorFlow, so you need TensorFlow Addons
from tensorflow_addons.activations import gelu

model_gelu = create_model(gelu)
print("Training model with GELU activation...")
history_gelu = model_gelu.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)
accuracy_gelu = model_gelu.evaluate(X_test, y_test, verbose=0)[1]
print(f"Test accuracy with GELU activation: {accuracy_gelu:.2f}")
