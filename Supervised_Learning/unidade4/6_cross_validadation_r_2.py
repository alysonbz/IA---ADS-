from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# Load the dataset
sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["radio"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFold object with k=6 and shuffling enabled
kf = KFold(n_splits=6, shuffle=True, random_state=5)

# Create the linear regression model
reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print cv_scores
print("Cross-validation scores:", cv_scores)

# Print the mean of cv_scores
print("Mean of cv_scores:", cv_scores.mean())

# Print the standard deviation of cv_scores
print("Standard deviation of cv_scores:", cv_scores.std())
