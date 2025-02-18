
from sklearn.preprocessing import StandardScaler
from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Import confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

scaler = StandardScaler()

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

X_norm = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=7)

# Fit the model to the training data
knn.fit(X_train, y_train)
# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
