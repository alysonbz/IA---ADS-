from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from sklearn.linear_model import LinearRegression


y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression().fit(X, y)

# Fit the model to the data


# Make predictions
predictions = reg.predict(X)

print(predictions[:5])