'''5. Disease Diagnosis  
Scenario: A healthcare provider uses patient data (symptoms, age, medical history) to diagnose diseases using K-NN.  
Question: What considerations should you take into account when choosing the value of $$ K $$? How would you ensure that the model is robust against overfitting?

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example Dataset
data = {
    "Symptom1": [1, 0, 1, 1, 0, 0, 1],
    "Symptom2": [2, 1, 3, 2, 1, 0, 3],
    "Age": [25, 34, 23, 45, 50, 36, 29],
    "History": [1, 0, 1, 0, 1, 0, 1],  # 1 = Chronic, 0 = No history
    "Disease": [1, 0, 1, 1, 0, 0, 1],  # 1 = Diagnosed, 0 = Not diagnosed
}
df = pd.DataFrame(data)

# Features and Target
X = df[["Symptom1", "Symptom2", "Age", "History"]]
y = df["Disease"]

# Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Finding Optimal K
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    accuracies.append(scores.mean())

# Plot Accuracy vs. K
plt.plot(k_values, accuracies, marker='o')
plt.title("Accuracy vs. K Value")
plt.xlabel("K")
plt.ylabel("Cross-Validated Accuracy")
plt.xticks(k_values)
plt.grid()
plt.show()

# Choosing Best K
optimal_k = k_values[accuracies.index(max(accuracies))]
print(f"Optimal K: {optimal_k}")

# Train Final Model with Optimal K
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train, y_train)
accuracy = final_knn.score(X_test, y_test)
print(f"Test Set Accuracy with K={optimal_k}: {accuracy:.2f}")
