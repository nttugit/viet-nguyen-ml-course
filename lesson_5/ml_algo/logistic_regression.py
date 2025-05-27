import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Example data: hours studied vs pass/fai
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]) # Study hours
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 0: fail

# Train
model  = LogisticRegression()
model.fit(X,y)


X_test = np.linspace(0,11,10).reshape(-1,1)
X_test_100 = np.linspace(0,11,100).reshape(-1,1)

# Predict
y_predict = model.predict(X_test)
y_predict_prob = model.predict_proba(X_test)[:, 1] # only get passing (second column)
y_predict_prob_100 = model.predict_proba(X_test_100)[:, 1] # only get passing (second column)

print(y_predict)
print(y_predict_prob)
print(f'3.5 study hour prediction: {model.predict([[3.5]])}')
print(f'3.5 study hour prediction (probability): {model.predict_proba([[3.5]])}')
print(f'6.5 study hour prediction: {model.predict([[6.5]])}')
print(f'6.5 study hour prediction (probability): {model.predict_proba([[6.5]])}')

# Visualization 

# Original
fig, axes = plt.subplots(1,3,figsize=(12,6))
axes[0].scatter(X,y,color='blue',label='Actual data')
axes[0].plot(X_test,y_predict,color='red',label='Predicted data')
axes[0].set_xlabel('Hours studied')
axes[0].set_ylabel('Pass/Fail')
# axes[0].set_title('Exam Result Prediction with Logistic Regression')

# Probability
axes[1].scatter(X,y,color='blue',label='Actual data')
axes[1].plot(X_test,y_predict_prob,color='red',label='Predicted data')
axes[1].set_xlabel('Hours studied')
axes[1].set_ylabel('Pass/Fail')
axes[1].set_title('Exam Result Predictiob with Logistic Regression')

# Look smoother with more data points
axes[2].scatter(X,y,color='blue',label='Actual data')
axes[2].plot(X_test_100,y_predict_prob_100,color='red',label='Predicted data')
axes[2].set_xlabel('Hours studied')
axes[2].set_ylabel('Pass/Fail')
# axes[2].set_title('Exam Result Prediction (Probability) with Logistic Regression')

plt.tight_layout()
plt.legend()
plt.show()