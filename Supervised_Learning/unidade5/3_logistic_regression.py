from sklearn.metrics import confusion_matrix, classification_report

from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#Import LogisticRegression
from sklearn.linear_model import LogisticRegression

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

# Instantiate the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]
y_pred = logreg.predict(X_test)

print(y_pred_probs[:10])
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))