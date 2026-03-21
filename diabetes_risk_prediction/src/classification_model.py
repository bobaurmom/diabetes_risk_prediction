import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ── Load Data ──────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, '..', 'data', 'clean_diabetes.csv'))
 
X = df.drop('class', axis=1)
y = df['class']
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# ── Random Forest ──────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
 
print("\n========== Random Forest ==========")
print(f"Test Accuracy : {accuracy_score(y_test, rf_pred):.4f}")
print(f"CV Accuracy   : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['Negative', 'Positive']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
joblib.dump(rf, 'random_forest_model.pkl')
model = joblib.load('random_forest_model.pkl')
new_patient = pd.DataFrame([[45, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]])
