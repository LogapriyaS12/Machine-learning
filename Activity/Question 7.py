'''7. Credit Risk Assessment  
Scenario: A bank uses logistic regression to determine the probability of a loan applicant defaulting on their loan based on their credit score, income level, and employment status.  
Question: How would you interpret the coefficients of your logistic regression model in this context? What implications do these coefficients have for risk assessment?'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Example Dataset
data = {
    "Credit_Score": [750, 680, 720, 640, 800, 600, 710],
    "Income_Level": [50000, 40000, 55000, 30000, 80000, 20000, 45000],
    "Employment_Status": [1, 0, 1, 0, 1, 0, 1],  # 1 = Employed, 0 = Unemployed
    "Default": [0, 1, 0, 1, 0, 1, 0],  # 1 = Default, 0 = No Default
}
df = pd.DataFrame(data)

# Features and Target
X = df[["Credit_Score", "Income_Level", "Employment_Status"]]
y = df["Default"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Coefficients Interpretation
coefficients = model.coef_[0]
intercept = model.intercept_[0]
feature_names = X.columns

print("Intercept:", intercept)
print("Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Probability of Default for a Sample Applicant
sample_applicant = np.array([[700, 45000, 1]])  # Example input
probability_default = model.predict_proba(sample_applicant)[0, 1]
print(f"Probability of Default for Sample Applicant: {probability_default:.2f}")
