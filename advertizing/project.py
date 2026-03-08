# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
data = pd.read_csv("advertising.csv")

# Display first 5 rows
print(data.head())

# Features (Advertising platforms)
X = data[['TV', 'Radio', 'Newspaper']]

# Target variable (Sales)
y = data['Sales']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# Print model coefficients
print("Model Coefficients:", model.coef_)

# Model evaluation
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Example prediction
new_data = [[230, 37, 69]]  # Example advertising budget
predicted_sales = model.predict(new_data)

print("Predicted Sales:", predicted_sales)
