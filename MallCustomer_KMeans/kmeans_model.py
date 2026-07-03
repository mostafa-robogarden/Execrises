import matplotlib.pyplot as plt
import numpy as np

from preporcessing import preprocess
class KMeansClustering:
    def __init__(self, k, tol=1e-4, max_iters=100):
        self.tol = tol
        self.max_iters = max_iters
        self.k = k
    def EuclideanDistance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    def fit(self, X):
        np.random.seed(42)
        centroids  = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for iteration in range(self.max_iters):
            clusters = np.array([np.argmin(np.linalg.norm(x - centroids, axis=1)) for x in X])
            #clusters = np.array([np.argmin([self.EuclideanDistance(x, centroids) for x in X])])
            new_centroids = np.array([X[clusters == i].mean(axis = 0) for i in range(self.k)])
            self.plot(new_centroids, X, clusters, iteration)
            if(np.all(np.linalg.norm(centroids - new_centroids) < self.tol)):
            #if np.all(self.EuclideanDistance(centroids, new_centroids) < self.tol):
                break
            centroids = np.array(new_centroids)
    def plot(self, new_centroids, X, clusters, iteration):
        plt.figure(figsize=(6, 5))
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolors='k', alpha=0.6)
        plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Centroids')
        plt.title(f"Iteration {iteration+1}")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    X = preprocess()
    kmeans = KMeansClustering(k=3, tol=1e-4, max_iters=100)
    kmeans.fit(X)
        