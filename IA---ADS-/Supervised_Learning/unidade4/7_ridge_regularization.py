from src.utils import load_sales_clean_dataset
from sklearn.model_selection import train_test_split
# Import Ridge
from ____.____ import ____


sales_df = load_sales_clean_dataset()
# Create X and y arrays
X = sales_df.drop(["sales","influencer"], axis=1)
y = sales_df["sales"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
    # Create a Ridge regression model
    ridge = ____

    # Fit the data
    ____

    # Obtain R-squared
    score = ____
    ridge_scores.append(score)
print(ridge_scores)