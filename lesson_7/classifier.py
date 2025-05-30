import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# Import data
data = pd.read_csv("../datasets/diabetes.csv")

# Split data (features, target)
target_col = "Outcome"
X = data.drop(target_col,axis=1)
y = data[target_col]

# Split data (train, test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scale data (preprocessing)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Init model and train
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
report = classification_report(y_test, y_pred)
print(report)

cm  =confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.savefig("diabetes.png")
plt.show()

