import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
red_df = pd.read_csv('data/winequality-red.csv', sep=';')
white_df = pd.read_csv('data/winequality-white.csv', sep=';')

# Combine the datasets
wine_df = pd.concat([red_df, white_df], axis=0).reset_index(drop=True)

# Separate features and target variable
X = wine_df.drop(columns=['quality'])
y = wine_df['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
preds = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, preds):.4f}")
print(f"R2 Score: {r2_score(y_test, preds):.4f}")

# Save the model
joblib.dump(model, 'wine_quality_model.pkl')