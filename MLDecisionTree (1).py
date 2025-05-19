import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. Load and Prepare the Iris Dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# training and testing
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

clf_iris = DecisionTreeClassifier(max_depth=5)  # Limit depth to 5 to avoid a too deep tree
clf_iris.fit(X_train_iris, y_train_iris)

# Evaluate the test
y_pred_iris = clf_iris.predict(X_test_iris)
accuracy_iris = metrics.accuracy_score(y_test_iris, y_pred_iris)
print(f"Accuracy on Iris dataset: {accuracy_iris * 100:.2f}%")

# Plot the decision tree
plt.figure(figsize=(15,10))
plot_tree(clf_iris, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

X_pima = pima.drop('label', axis=1)
y_pima = pima['label']

# training and testing datasets
X_train_pima, X_test_pima, y_train_pima, y_test_pima = train_test_split(X_pima, y_pima, test_size=0.3, random_state=42)

# Decision Treemodel
clf_pima = DecisionTreeClassifier(max_depth=5)  # Limit depth to 5 to avoid a too deep tree
clf_pima.fit(X_train_pima, y_train_pima)

# Evaluate
y_pred_pima = clf_pima.predict(X_test_pima)
accuracy_pima = metrics.accuracy_score(y_test_pima, y_pred_pima)
print(f"Accuracy on Diabetes dataset: {accuracy_pima * 100:.2f}%")

# Plot the decision tree
plt.figure(figsize=(15,10))
plot_tree(clf_pima, filled=True, feature_names=X_pima.columns, class_names=["Negative", "Positive"], rounded=True)
plt.title("Decision Tree")
plt.show()
