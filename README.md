# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required libraries, load the Iris dataset, and create a DataFrame containing feature values and target labels.
2. Separate the dataset into input features (X) and target values (y), then split the data into training and testing sets.
3. Create and train the SGDClassifier model using the training data and predict the output for the test data.
4. Calculate the accuracy of the model, generate the confusion matrix, and visualize the results using a heatmap graph.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Karthi V
RegisterNumber:  212225230130
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


```

## Output:

<img width="720" height="744" alt="image" src="https://github.com/user-attachments/assets/dfbc5ee5-34be-4273-b44b-7b67f46cd0c7" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
