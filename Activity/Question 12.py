'''12. Predicting Credit Card Fraud
Scenario: A financial institution is using logistic regression to identify fraudulent transactions based on transaction amount, location, and user behavior.  
Question: In setting up your data pipeline, how would you ensure that the model is trained on balanced classes? What techniques could you employ to address class imbalance?'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

data = pd.read_csv('credit_card_data.csv')
X = data[['transaction_amount', 'location', 'user_behavior']]
y = data['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
