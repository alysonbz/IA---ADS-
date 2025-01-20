from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split

#Import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values


#Normalize the data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)


#Split the data
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42,stratify=y)


# Instantiate the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)
y_preb = logreg.predict(X_test)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print(y_pred_probs[:10])
print(confusion_matrix(y_test, y_preb))
print(classification_report(y_test, y_preb))
