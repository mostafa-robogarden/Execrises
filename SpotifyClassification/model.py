import numpy as np
from sklearn.model_selection import train_test_split
import joblib

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

data = joblib.load("spotify_scaled_data.pkl")
X, y = data["X"], data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = len(np.unique(y))
epochs = 1000
lr = 0.01
# One-hot encode y
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_oh = one_hot(y_train, output_dim)
y_test_oh = one_hot(y_test, output_dim)
# Initialize weights
#y = WX + b
np.random.seed(0)
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))
weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
# Training
for epoch in range(epochs):
    # Forward
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    # Loss (MSE)
    loss = np.mean((y_train_oh - a2) ** 2)
    # Backprop
    d2 = (y_train_oh - a2) * sigmoid_derivative(a2)
    dW2 = np.dot(a1.T, d2)
    db2 = np.sum(d2, axis=0, keepdims=True)
    d1 = np.dot(d2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X_train.T, d1)
    db1 = np.sum(d1, axis=0, keepdims=True)
    # Update
    W1 += lr * dW1
    b1 += lr * db1
    W2 += lr * dW2
    b2 += lr * db2
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    joblib.dump(weights, "manual_ann_weights.pkl")