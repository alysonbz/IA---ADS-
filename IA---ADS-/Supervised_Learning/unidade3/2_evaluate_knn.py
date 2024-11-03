from sklearn.neighbors import KNeighborsClassifier
from src.utils import load_churn_dataset

# Import the module
____

churn_df = load_churn_dataset()
X = churn_df[["account_length",  "total_day_charge" , "total_eve_charge",  "total_night_charge","total_intl_charge","number_customer_service_calls"]].values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = ____(____, ____, test_size=____, random_state=____, stratify=____)

knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
____

# Print the accuracy
print(knn.score(____, ____))