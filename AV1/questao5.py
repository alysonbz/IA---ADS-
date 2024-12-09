import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

tesla = pd.read_csv("dataset/regression/Tesla.csv")
ferrari = pd.read_csv("dataset/regression/Ferrari.csv")
ferrari24 = pd.read_csv("dataset/regression/Ferrari24.csv")
tesla24 = pd.read_csv("dataset/regression/Tesla24.csv")
datasets = [tesla, ferrari, ferrari24, tesla24]
for i, df in enumerate(datasets):
    df["Source"] = f"Dataset_{i+1}"  # Add a source column for tracking
merged_data = pd.concat(datasets, ignore_index=True)

# drop non-numeric or irrelevant columns (e.g., 'Date', 'Source')
merged_data = merged_data.drop(["Date", "Source"], axis=1)
merged_data = merged_data.dropna()
X = merged_data.drop("Close", axis=1)  # Features
y = merged_data["Close"].values        # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso Regression
lasso = Lasso(alpha=0.1)  # Adjust alpha as needed
lasso.fit(X_train, y_train)

# feature importance
lasso_coef = lasso.coef_
features = X.columns

print("Lasso Coefficients:\n", dict(zip(features, lasso_coef)))

plt.figure(figsize=(10, 6))
plt.bar(features, lasso_coef, color='blue')
plt.xlabel("Features")
plt.ylabel("Importance (Lasso Coefficients)")
plt.title("Feature Importance Determined by Lasso Regression")
plt.xticks(rotation=45)
plt.grid()
plt.show()

merged_data = merged_data.drop("Volume", axis=1)
merged_data.to_csv("dataset/regression/merged_ferrari_tesla.csv", index=False)


