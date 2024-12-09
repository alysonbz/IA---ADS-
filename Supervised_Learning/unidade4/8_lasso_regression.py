import matplotlib.pyplot as plt
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import Lasso  # Import Lasso

# Load the cleaned dataset
sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df.drop(["sales", "influencer"], axis=1)
y = sales_df["sales"].values
sales_columns = X.columns  # Save feature names for visualization

# Instantiate a Lasso regression model with alpha=0.3
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print("Lasso Coefficients:", lasso_coef)

# Visualize the coefficients using a bar chart
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.ylabel("Coefficient Value")
plt.title("Lasso Coefficients")
plt.show()
