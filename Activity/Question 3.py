'''3. Evaluating Marketing Strategies  
Scenario: A marketing team has created a decision tree to evaluate different advertising strategies for a new product launch.  
Question: How would you assess the effectiveness of the current decision tree structure? What specific metrics or outcomes would you analyze to determine if any adjustments are necessary?'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Sample Data
data = {
    "Ad_Budget": [1000, 1500, 800, 2000, 500],
    "Target_Age": [25, 35, 20, 40, 30],
    "Medium": ["Online", "TV", "Online", "Print", "Online"],
    "Success": [1, 1, 0, 1, 0],
}

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.DataFrame(data)

# Features and Target
X = df[["Ad_Budget", "Target_Age", "Medium"]]
y = df["Success"]

# Preprocessing for Categorical Data
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), ["Medium"])], remainder="passthrough"
)
X = preprocessor.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Predictions
y_pred = tree.predict(X_test)

# Evaluation Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-Validation Score
cv_score = cross_val_score(tree, X, y, cv=5)
print(f"\nCross-Validation Score: {cv_score.mean()}")
