import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("cricket_data.csv")  # Assume columns: ['overs', 'wickets', 'runs', 'total_score']
X = df[['overs', 'wickets', 'runs']]
y = df['total_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict total score
new_data = [[15, 3, 90]]  # 15 overs, 3 wickets lost, 90 runs scored so far
predicted_score = model.predict(new_data)
print(f"Predicted total score: {predicted_score[0]}")
