import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Sample Dataset (Replace with actual dataset)
data = {
    'Location': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'Size': [1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000, 3200],
    'Bedrooms': [2, 3, 3, 4, 4, 5, 5, 5, 6],
    'Price': [300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000]
}
df = pd.DataFrame(data)

# Features and Target
X = df[['Location', 'Size', 'Bedrooms']]
y = df['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R-squared: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
