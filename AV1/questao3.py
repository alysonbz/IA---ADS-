from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from AV1.src.utils import load_water_quality
from sklearn.metrics import accuracy_score
import numpy as np

waterQuality = load_water_quality()

X = waterQuality.drop(['is_safe'], axis=1).values
y = waterQuality['is_safe'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

def knn_mahalanobis(X_train, X_test, y_train, y_test, n_neighbors=7):
    cov_matrix = np.cov(X_train.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='mahalanobis', metric_params={'VI': inv_cov_matrix})

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

accuracy_log = knn_mahalanobis(X_train_log, X_test_log, y_train, y_test)
accuracy_norm = knn_mahalanobis(X_train_norm, X_test_norm, y_train, y_test)

print(f"Acurácia com normalização logarítmica: {accuracy_log * 100}%")
print(f"Acurácia com média zero e variância unitária: {accuracy_norm * 100}%")
