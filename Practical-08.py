import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Sample dataset: Text documents and their corresponding classes
# You can replace this with your actual dataset
data = {
    'text': [
        'I love programming',
        'Python is great',
        'Java is awesome for OOP',
        'I hate bugs in code',
        'JavaScript is fun to learn',
        'Debugging is important',
        'Python makes data science easier',
        'I dislike poor documentation'
    ],
    'class': ['positive', 'positive', 'positive', 'negative', 'positive', 'neutral', 'positive', 'negative']
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and labels (y)
X = df['text']
y = df['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into numerical form using CountVectorizer (Bag of Words)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test_counts)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y))
recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y))

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
