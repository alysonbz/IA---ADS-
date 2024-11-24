import numpy as np
import pandas as pd

from src.utils import load_sales_clean_dataset
sales_df = load_sales_clean_dataset()

# Create X from the radio column's values
X = sales_df.drop("radio", axis=1).values

# Create y from the sales column's values
y = sales_df["radio"].values

# Reshape X
X = X[:, 3]
print(y.shape, X.shape)
X = X.reshape(-1, 1)

# Check the shape of the features and targets
print(X_radio.shape)