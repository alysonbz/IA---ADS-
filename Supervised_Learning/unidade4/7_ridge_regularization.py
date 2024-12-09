from src.utils import load_sales_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Load the cleaned dataset
sales_df = load_sales_clean_dataset()

# Separate features (X) and target (y)
X = sales_df.drop(["sales", "influencer"], axis=1)
y = sales_df["sales"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# List of alpha values to test
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []

# Loop to fit the model with different alpha values
for alpha in alphas:
    # Create the Ridge regression model
    ridge = Ridge(alpha=alpha)
    
    # Fit the model to the training data
    ridge.fit(X_train, y_train)
    
    # Obtain the R-squared score on the test data
    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)

# Print the R-squared scores for each alpha value
print("Ridge Scores (R²) for different alpha values:", ridge_scores)
