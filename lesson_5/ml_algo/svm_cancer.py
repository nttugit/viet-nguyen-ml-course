from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()
X = data.data # features
y = data.target # Labels: 0 = malignant, 1 = benign
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(f'X train shape: {X_train.shape}')
print(f'Y test shape: {y_test.shape}')

# Train
model = SVC(kernel='linear', C=1) # kerkel ='rbf' or 'poly'
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_predict)
cfm = confusion_matrix(y_test,y_predict)
report = classification_report(y_test,y_predict, target_names=data.target_names)

print(f"Accuracy: {accuracy:.2f}")
print(f"Classification report\n: {report}")

# Visualize
plt.figure(figsize=(6,5))
sns.heatmap(data=cfm,annot=True,fmt='d',cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("Breast cancer classification using SVM - Confusion matrix")
plt.tight_layout()
plt.show()