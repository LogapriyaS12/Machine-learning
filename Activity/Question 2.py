'''2. Time Series Data  
Scenario: You have monthly data on electricity consumption over several years and want to predict future consumption based on trends and seasonal patterns.  
Question: Can linear regression be effectively used in this scenario? If so, how would you incorporate time as a variable in your model?'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = {
    "Month": pd.date_range(start="2015-01-01", periods=60, freq="M"),
    "Consumption": [120, 125, 130, 128, 122, 121, 135, 140, 145, 142, 130, 125] * 5,
}

df = pd.DataFrame(data)


df["Time_Index"] = np.arange(len(df))  # Trend feature
df["Month_Sin"] = np.sin(2 * np.pi * df["Time_Index"] / 12)  # Seasonality
df["Month_Cos"] = np.cos(2 * np.pi * df["Time_Index"] / 12)  # Seasonality


X = df[["Time_Index", "Month_Sin", "Month_Cos"]]
y = df["Consumption"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)

# Results
print("Mean Squared Error:", mse)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

future_time_index = np.arange(len(df), len(df) + 12)
future_month_sin = np.sin(2 * np.pi * future_time_index / 12)
future_month_cos = np.cos(2 * np.pi * future_time_index / 12)
future_X = pd.DataFrame({
    "Time_Index": future_time_index,
    "Month_Sin": future_month_sin,
    "Month_Cos": future_month_cos,
})
future_predictions = model.predict(future_X)

print("Future Predictions:", future_predictions)
