# Linear regression with outliers

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2],[3],[4],[5]])
y = np.array([30000, 35000, 50000, 55000, 60000])

X_outlier = np.array([[1], [2],[3],[4],[5],[2]])
y_outlier = np.array([30000, 35000, 50000, 55000, 60000, 80000])

model = LinearRegression()
model.fit(X,y)

model2 = LinearRegression()
model2.fit(X_outlier,y_outlier)

y_pred = model.predict(X)
y_pred_outlier = model2.predict(X_outlier)

print(f'Slope: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')

prediction_1 = model.predict([[2.5]])
print(prediction_1)

fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].scatter(X,y,color='blue',label='Actual label')
axes[0].plot(X,y_pred, color='red',label='Predicted label')
axes[0].set_xlabel('Years of experience')
axes[0].set_ylabel('Salary')
axes[0].set_title('Salary Prediction using Linear Regression')

axes[1].scatter(X_outlier,y_outlier,color='blue',label='Actual label')
axes[1].plot(X_outlier,y_pred_outlier, color='red',label='Predicted label')
axes[1].set_xlabel('Years of experience')
axes[1].set_ylabel('Salary')
axes[1].set_title('Salary Prediction using Linear Regression with Outliers')

plt.tight_layout()
plt.show()