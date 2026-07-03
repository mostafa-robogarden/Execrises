import numpy as np
import pandas as pd
import os

def load_data():
    data_list = []
    DIR = 'dataset'
    for filename in ['Train.csv', 'Valid.csv', 'Test.csv']:
        df = pd.read_csv(os.path.join(DIR, filename), index_col=None, header=0)
        data_list.append(df)
    return pd.concat(data_list, axis=0, ignore_index=True)

def text_to_vector(corpus):
    vocab = {}
    for text in corpus:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    vectors = np.zeros((len(corpus), len(vocab)))
    for i, text in enumerate(corpus):
        for word in text.split():
            if word in vocab:
                vectors[i, vocab[word]] += 1
    
    return vectors, vocab

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    for _ in range(epochs):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)
        
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        weights -= lr * dw
        bias -= lr * db
    return weights, bias

def predict(X, weights, bias):
    return (sigmoid(np.dot(X, weights) + bias) >= 0.5).astype(int)

data = load_data()
x = data["text"]
y = data["label"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train_vec, vocab = text_to_vector(x_train)
x_test_vec, _ = text_to_vector(x_test)

weights, bias = train_logistic_regression(x_train_vec, y_train)
y_pred = predict(x_test_vec, weights, bias)

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Predict new review
new_review = ["I had fun"]
new_review_vec, _ = text_to_vector(new_review)
pred_sentiment = predict(new_review_vec, weights, bias)
if pred_sentiment[0] == 0:
    print("Review is negative")
else:
    print("Review is positive")