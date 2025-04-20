from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

for n in [10, 50, 100]:
    clf = RandomForestClassifier(n_estimators=n).fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"Trees: {n} | Accuracy: {accuracy_score(y_test, preds):.2f}")
