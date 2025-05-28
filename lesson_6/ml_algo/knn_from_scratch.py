
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Tính khoảng cách đến tất cả điểm dữ liệu
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Lấy index các k điểm gần nhất
        k_indices = np.argsort(distances)[:self.k]
        # Từ indices trên, lấy labels của k điểm gần nhất đó
        k_neighbor_labels = [self.y_train[index] for index in k_indices]
        # Majority vote
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0] 


# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNN(5)
knn.fit(X_train,y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# Predict a single point
x = np.zeros((30,))
predict = knn.predict(np.array([x]))
print("Result: ", predict)