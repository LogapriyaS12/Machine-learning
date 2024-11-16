'''11.Weather Prediction  
Scenario: You create a simple linear regression model to predict daily temperatures based on historical weather data.  
Question: If your predictions are consistently inaccurate because they do not capture seasonal variations, what does this indicate about bias? How might you improve your model's accuracy?'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('weather_data.csv')
data['day_of_year'] = pd.to_datetime(data['date']).dt.dayofyear
data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
X = data[['sin_day', 'cos_day']]
y = data['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
