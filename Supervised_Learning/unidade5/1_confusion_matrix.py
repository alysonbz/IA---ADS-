from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Import confusion matrix
____

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
____

# Predict the labels of the test data: y_pred
y_pred = ____

# Generate the confusion matrix and classification report
print(____(____, ____))
print(____(____, ____))
