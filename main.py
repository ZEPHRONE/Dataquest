import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

# Load the CSV dataset (replace 'mastertrain.csv' with your CSV file)
df = pd.read_csv('mastertrain.csv')

# Split the data into features (X) and targets (y1 for target1 and y2 for target2)
X = df.drop(['Target1', 'Target2'], axis=1)
X = pd.get_dummies(X, columns=['Feature3', 'Feature4', 'Gender2', 'Customer Type', 'Type of Travel', 'Class'])
y1 = df['Target1']
y2 = df['Target2']

# Identify non-numeric columns and exclude them from the feature set
non_numeric_cols = ['Dog Name']
numeric_cols = [col for col in X.columns if col not in non_numeric_cols]
X_numeric = X[numeric_cols]

# Split the data into training and testing sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X_numeric, y1, y2, test_size=0.2, random_state=42)

# Create an imputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on training data
X_train = imputer.fit_transform(X_train)

# Transform the testing data using the same imputer
X_test = imputer.transform(X_test)

# Create and train the Random Forest model for target1
model_target1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_target1.fit(X_train, y1_train)

# Create and train the Random Forest model for target2
model_target2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_target2.fit(X_train, y2_train)

# Evaluate the model for target1
y1_pred = model_target1.predict(X_test)
mse_target1 = mean_squared_error(y1_test, y1_pred)
print(f'MSE for target1: {mse_target1}')

# Evaluate the model for target2
y2_pred = model_target2.predict(X_test)
mse_target2 = mean_squared_error(y2_test, y2_pred)
print(f'MSE for target2: {mse_target2}')

# Load the new data (replace 'mastertest.csv' with your new data file)
new_data = pd.read_csv('mastertest.csv')

# Perform one-hot encoding on the new data and align columns with training data
new_data_encoded = pd.get_dummies(new_data, columns=['Feature3', 'Feature4', 'Gender2', 'Customer Type', 'Type of Travel', 'Class'])
new_data_encoded = new_data_encoded.reindex(columns=X_numeric.columns, fill_value=0)

# Transform the new data using the same imputer
new_data_encoded = imputer.transform(new_data_encoded)

# Make predictions for target1 on the new data
predictions_target1 = model_target1.predict(new_data_encoded)
# Apply a threshold to round predictions to 0 or 1
predictions_target1 = np.round(predictions_target1).astype(int)

# Make predictions for target2 on the new data
predictions_target2 = model_target2.predict(new_data_encoded)
# Apply a threshold to round predictions to 0 or 1
predictions_target2 = np.round(predictions_target2).astype(int)

# Now 'predictions_target1' and 'predictions_target2' contain your model's binary predictions for the new data (0 or 1)
# You can save these predictions to a CSV file or use them as needed.

# Save predictions to a CSV file (replace 'predictions.csv' with your desired filename)
predictions_df = pd.DataFrame({'Predicted_Target1': predictions_target1, 'Predicted_Target2': predictions_target2})
predictions_df.to_csv('predictions.csv', index=False)
# Make predictions for target1 on the new data
predictions_target1 = model_target1.predict(new_data_encoded)
# Convert predictions to 0 or 1
predictions_target1 = (predictions_target1 > 0.5).astype(int)

# Make predictions for target2 on the new data
predictions_target2 = model_target2.predict(new_data_encoded)
# Convert predictions to 0 or 1
predictions_target2 = (predictions_target2 > 0.5).astype(int)


# Optionally, you can print or further process the predictions
print('Predictions for Target1:')
print(predictions_target1)

print('Predictions for Target2:')
print(predictions_target2)
