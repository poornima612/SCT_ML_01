import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Example dataset (replace with real data or load from CSV)
data = {
    'square_feet': [1500, 1600, 1700, 1800, 1900],
    'bedrooms': [3, 3, 4, 4, 5],
    'bathrooms': [2, 2, 3, 3, 4],
    'price': [300000, 320000, 340000, 360000, 400000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
