from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from ____.____ import ____


y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = ____()

# Fit the model to the data
____

# Make predictions
predictions = ____

print(__)