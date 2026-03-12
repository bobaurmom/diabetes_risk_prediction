from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib 

class LogisticModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = LogisticRegression(max_iter=1000)  # Increased max_iter
    def train(self):
        print("training logistic regression model")
        self.model.fit(self.X_train, self.y_train) #x_train is the training data and y_train is the target variable

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test) #y_pred is the predicted target
        print("Prediction done")

        # we compare between actual and predicted values
        acc = accuracy_score(y_test, y_pred)   # accuracy tell us the proportion of correct predictions out of total predictions
        prec = precision_score(y_test, y_pred) # precision tells us the proportion of true positive predictions out of all positive predictions
        matrix = confusion_matrix(y_test, y_pred) # confusion matrix is a table that is used to evaluate the performance of a classification model, it shows the number of true positives, true negatives, false positives, and false negatives
        recall = matrix[1][1] / (matrix[1][1] + matrix[1][0])  # recall tells us the proportion of true positive predictions out of all actual positive cases

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{matrix}")
        print(f"True Positive: {matrix[1][1]}")
        print(f"True Negative: {matrix[0][0]}")
        print(f"False Positive: {matrix[0][1]}")
        print(f"False Negative: {matrix[1][0]}")

        return y_pred
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        print("job done")


# Example usage:

df = pd.read_csv('data/clean_diabetes2.csv') 
# Split Features and Target
X = df.drop('class', axis=1)
y = df['class']

# Split Data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Evaluate
l1 = LogisticModel(X_train, y_train)
l1.train()
l1.predict(X_test, y_test)


        
# Save results
l1.save_model('models/logistic_model.joblib')
