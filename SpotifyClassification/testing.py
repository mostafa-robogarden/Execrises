import joblib
import numpy as np
from sklearn.metrics import accuracy_score

weights = joblib.load("manual_ann_weights.pkl")
W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)
def predict(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    output = softmax(z2)
    return np.argmax(output, axis=1)

data = joblib.load("spotify_scaled_data.pkl")
X = data["X"]
y = data["y"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = predict(X_test)
print("Manual ANN Accuracy:", accuracy_score(y_test, y_pred))
sample_song = [0.0202, 0.266, 349667, 0.348, 0.664, 10,0.16, -11.609, 0, 0.0371, 144.154, 4.0, 0.393, 1]
scaler = joblib.load("scaler.pkl")
sample_scaled = scaler.transform([sample_song])
output = predict(sample_scaled)
label = np.argmax(output)
print("Predicted class:", label)
print("Confidence:", output[0])