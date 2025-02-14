from src.utils import load_sales_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

sales_df = load_sales_clean_dataset()
# Create X and y arrays
X = sales_df.drop(["sales", "influencer"], axis=1)
y = sales_df["sales"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
    # Create a Ridge regression model
    ridge = Ridge(alpha=alpha)
    # Fit the data
    ridge.fit(X_train, y_train)
    # Obtain R-squared
    score = ridge.predict(X_test)
    ridge_scores.append(ridge.score(X_test, y_test))
print(ridge_scores)