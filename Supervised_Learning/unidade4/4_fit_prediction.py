from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sales_df = load_sales_clean_dataset()

# Import mean_squared_error
from ____.____ import ____

# Create X and y arrays
X = sales_df.____(["____","___" ],axis=____)
y = sales_df["____"].____

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = ____

# Fit the model to the data
____

# Make predictions
y_pred = reg.____(____)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Compute R-squared
r_squared = reg.____(____, ____)

# Compute RMSE
rmse = ____(____, ____, squared=____)

# Print the metrics
print("R^2: {}".format(____))
print("RMSE: {}".format(____))