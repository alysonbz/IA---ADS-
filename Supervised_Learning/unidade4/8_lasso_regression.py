import matplotlib.pyplot as plt
from src.utils import load_sales_clean_dataset

# Import Lasso
from sklearn.linear_model import Lasso

sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df.drop(["sales","influencer"], axis=1)
y = sales_df["sales"].values
sales_columns = X.columns

# Instantiate a lasso regression model
lasso = Lasso()

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
