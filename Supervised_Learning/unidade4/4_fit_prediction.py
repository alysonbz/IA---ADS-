import numpy as np

from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
sales_df = load_sales_clean_dataset()
from sklearn.metrics import root_mean_squared_error

# Import mean_squared_error

# Create X and y arrays
X = sales_df.drop(["sales","influencer"],axis=1)
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Compute R-squared
r_squared = reg.predict(X_test)

# Compute RMSE
# rmse = (, , squared=)
rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))