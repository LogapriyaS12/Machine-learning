'''4. Financial Investment Decisions  
Scenario: An investment firm uses a decision tree to analyze potential investment opportunities based on market trends and economic indicators.  
Question: What process would you follow to update the decision tree as market conditions change? How would you visualize these changes for clarity among team members?'''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 1: Updated Dataset
data = {
    "Market_Trend": [1, 0, 1, 1, 0, 0, 1],  # 1 = Positive, 0 = Negative
    "Economic_Indicator": [2.5, 1.2, 3.1, 2.8, 1.0, 0.8, 3.0],  # GDP Growth
    "Investment_Return": [1, 0, 1, 1, 0, 0, 1],  # 1 = Success, 0 = Failure
}
df = pd.DataFrame(data)

# Features and Target
X = df[["Market_Trend", "Economic_Indicator"]]
y = df["Investment_Return"]

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Visualize Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(tree, feature_names=["Market_Trend", "Economic_Indicator"], class_names=["Failure", "Success"], filled=True)
plt.title("Updated Decision Tree")
plt.show()

# Visualize Feature Importance
feature_importance = tree.feature_importances_
plt.bar(["Market_Trend", "Economic_Indicator"], feature_importance, color='skyblue')
plt.title("Feature Importance Comparison")
plt.ylabel("Importance")
plt.show()
\
