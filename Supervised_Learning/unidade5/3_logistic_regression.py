from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
#Import LogisticRegression
____


diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)


# Instantiate the model
logreg = ____

# Fit the model
____

# Predict probabilities
y_pred_probs = logreg.____(____)[____, ____]

print(y_pred_probs[:10])
