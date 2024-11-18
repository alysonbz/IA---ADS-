from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from sklearn.linear_model import LinearRegression


y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])