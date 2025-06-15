import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

with np.load("../datasets/mnist.npz") as f:
    x_train = f['x_train']
    x_test = f['x_test']
    y_train = f['y_train']
    y_test = f['y_test']

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] * x_train.shape[2])
x_train = np.reshape(x_train, (x_train.shape[0],-1))
# x_train = x_train.reshape(x_train.shape[0],-1)
x_test = np.reshape(x_test, (x_test.shape[0],-1))
# print(x_train.shape)

cls = RandomForestClassifier()
cls.fit(x_train, y_train)

y_pred = cls.predict(x_test)
result = classification_report(y_test, y_pred)
print(result)
