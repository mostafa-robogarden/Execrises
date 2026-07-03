import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# Your preprocessing code
# =========================

DIR = 'dataset/PetImages'
IMG_SIZE = 100
categories = ['Dog', 'Cat']

def readData(directory):
    data = []
    for category in categories:
        path = os.path.join(directory, category)
        classification = categories.index(category)  # 0 for Dog, 1 for Cat
        file_list = os.listdir(path)
        for img in file_list:
            try:
                check_path = os.path.exists(os.path.join(path, img))
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    raise FileNotFoundError('image ' + img + ' not found in path ' + path)
                resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_img_array, classification])
            except Exception as exception:
                print('An exception occured with error: ' + str(exception))
                continue
    return data

def featureLabelSplit(data):
    X = []
    y = []
    for features, labels in data:
        X.append(features)
        y.append(labels)
    X_array = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # (N, 100, 100, 1)
    return X_array, y

def preprocessData(directory):
    data = readData(directory)
    random.shuffle(data)
    features, labels = featureLabelSplit(data)
    with open('features.pickle', 'wb') as file:
        pickle.dump(features, file)
    with open('labels.pickle', 'wb') as file:
        pickle.dump(labels, file)

# Run preprocessing ONCE (comment it out after first run if you want)
preprocessData(DIR)

# =========================
# Load preprocessed data
# =========================

with open('features.pickle', 'rb') as file:
    X = pickle.load(file)

with open('labels.pickle', 'rb') as file:
    y = pickle.load(file)

X = np.array(X, dtype='float32')
y = np.array(y, dtype='int32')

# Normalize pixel values from [0, 255] to [0, 1]
X = X / 255.0

print("Features shape:", X.shape)   # (N, 100, 100, 1)
print("Labels shape:", y.shape)     # (N,)

# =========================
# Train / val / test split
# =========================

# First: train+temp and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Then: train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val
)

print("Train size:", X_train.shape[0])
print("Val size:", X_val.shape[0])
print("Test size:", X_test.shape[0])

# =========================
# Build the CNN model
# =========================

model = models.Sequential([
    # Conv block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),

    # Conv block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Conv block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary output: Dog vs Cat
])

model.summary()

# =========================
# Compile the model
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train the model
# =========================

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# =========================
# Evaluate on test set
# =========================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# =========================
# (Optional) Plot training curves
# =========================

plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.show()