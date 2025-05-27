# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Step 3: Select numerical features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF']
target = 'SalePrice'

# Step 4: Handle missing values (basic approach)
train_df = train_df[features + [target]].dropna()
test_df = test_df[features].fillna(test_df[features].median())  # fill missing with median

# Step 5: Split the training data (for model evaluation)
X = train_df[features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_val)
print("Mean Squared Error (MSE):", mean_squared_error(y_val, y_pred))
print("R² Score:", r2_score(y_val, y_pred))

# Optional: Scatter plot of predicted vs actual
plt.scatter(y_val, y_pred, alpha=0.7)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted Sale Prices')
plt.grid(True)
plt.show()

# Step 8: Predict on test set
test_preds = model.predict(test_df)

# Step 9: Create submission file
submission = pd.DataFrame({
    'Id': sample_submission['Id'],
    'SalePrice': test_preds
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created successfully!")
