from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
# Import the necessary modules
from ____.____ import ____, ____

sales_df = load_sales_clean_dataset()
# Create X and y arrays
X = sales_df["radio"].values.reshape(-1, 1)
y = sales_df["sales"].values


#Create a KFold object
kf = ____(n_splits=____, shuffle=____, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = ____(____, ____, ____, cv=____)

# Print cv_scores
print(____)

# Print the mean
print(___(__))

# Print the standard deviation
print(___(__))

