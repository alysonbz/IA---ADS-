import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression


class KFold:

    def _init_(self, n_splits):
        # Initialize the number of splits for cross-validation
        self.n_splits = n_splits

    def _compute_score(self, X, y):
        # Compute the score for a given subset (to be implemented)
        return None

    def cross_val_score(self, model, X, y):
        scores = []

        # Part 1: Split the dataset X into n_splits folds
        fold_size = len(X) // self.n_splits
        for i in range(self.n_splits):
            # Define test and train indices
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            X_train = np.concatenate([X[:test_start], X[test_end:]])
            y_train = np.concatenate([y[:test_start], y[test_end:]])

            # Part 2: Train the model on the training set and calculate the score
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            # Append each score to the scores list
            scores.append(score)

        # Part 3: Return the list of scores
        return scores


# Load the cleaned dataset
sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFold object
kf = KFold(n_splits=6)

# Instantiate the linear regression model
reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = kf.cross_val_score(reg, X, y)

# Print the scores for each fold
print("Cross-Validation Scores:", cv_scores)

# Print the mean of the scores
print("Mean CV Score:", np.mean(cv_scores))

# Print the standard deviation of the scores
print("Standard Deviation of CV Scores:", np.std(cv_scores))
