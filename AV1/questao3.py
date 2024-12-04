from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.utils import load_new_dataframe_gender_classification
import numpy as np

gender = load_new_dataframe_gender_classification()

scaler = StandardScaler()

X = gender.drop(['gender'], axis=1)

offset = 1e-6
X_norm = scaler.fit_transform(X)
X_log = np.log(X + offset)

y = gender['gender'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)
X_train_log, X_test_log, y_train, y_test = train_test_split(X_log, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

print('score padrão', knn.score(X_test, y_test))

knn.fit(X_train_norm,y_train)

print('\nscore normalizado', knn.score(X_test_norm, y_test))

knn.fit(X_train_log,y_train)

print('\nscore log', knn.score(X_test_log, y_test))
