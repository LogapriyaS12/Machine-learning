'''1. Analyzing Student Performance:  
Scenario: You are analyzing factors that affect student performance in a standardized test. You collect data on study hours, attendance rates, and socioeconomic background.  
Question: How would you set up your linear regression model? What considerations would you make regarding the interpretation of coefficients in this context?'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Sample Data
data = {
    "Study_Hours": [10, 15, 8, 20, 5],
    "Attendance_Rate": [90, 85, 70, 95, 60],
    "Socioeconomic_Status": ["Low", "Medium", "High", "Low", "Medium"],
    "Test_Score": [85, 90, 78, 92, 65],
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[["Study_Hours", "Attendance_Rate", "Socioeconomic_Status"]]
y = df["Test_Score"]

# Preprocessing: One-Hot Encoding for Socioeconomic Status
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Study_Hours", "Attendance_Rate"]),
        ("cat", OneHotEncoder(), ["Socioeconomic_Status"]),
    ]
)

# Linear Regression Pipeline
model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Coefficients Interpretation
regressor = model.named_steps["regressor"]
preprocessor_fit = model.named_steps["preprocessor"]
encoded_features = preprocessor_fit.transformers_[1][1].get_feature_names_out(["Socioeconomic_Status"])
final_features = ["Study_Hours", "Attendance_Rate"] + list(encoded_features)
coefficients = regressor.coef_

for feature, coef in zip(final_features, coefficients):
    print(f"Coefficient for {feature}: {coef}")
