from sklearn.feature_extraction.text import HashingVectorizer

# Example 1
documents = ["cat", "dog", "mouse", "cat", "elephant"]

# Tao hashing vectorizer voi khong gian 8 chieu
vectorizer  =HashingVectorizer(n_features=2**3, alternate_sign=False)

# Chuyen van ban thanh dang vectorizer
X = vectorizer.transform(documents)
print(X)
print(X.toarray())

# Example 2
# Spam detection with hash encoding

# Sample dataset: messages and their labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB

texts = [
    "Win a free iPhone now",          # spam
    "Call this number to claim prize",# spam
    "Hey, are we still on for lunch?",# not spam
    "Don't forget the meeting today", # not spam
    "Congratulations! You won",       # spam
    "Let's catch up later"            # not spam
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = not spam

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Pipeline: HasingVectorizer + Naive Bayes classifier
model = make_pipeline(
    HashingVectorizer(n_features=32, alternate_sign=False),  # hash encoding
    MultinomialNB()
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_test)
print(y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))