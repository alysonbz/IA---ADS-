from src.utils import load_churn_dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

churn_df = load_churn_dataset()

# Create arrays for the features and the target variable
X = churn_df[["account_length", "number_customer_service_calls"]].values
y = churn_df["churn"].values

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))

X_test = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# Predict the labels for the X_teste
y_pred = knn.predict(X_test)

# Print the predictions for X_test
print("Predictions: {}".format(y_pred))