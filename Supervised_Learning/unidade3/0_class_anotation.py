from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

churn_df = load_churn_dataset()

# Obter atributos X e labels y
X = churn_df[["account_length", "number_customer_service_calls"]].values
y = churn_df["churn"].values

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Inicialize o classificador KNN.  Atribua 6 vizinhos.
knn = KNeighborsClassifier(n_neighbors=6)

# Treinar o modelo
knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))

# Execute uma predição com a função ``predict`` do ``knn``. Atribua como argumento ``X_test``
y_pred = knn.predict(X_test)

# Print as predições realizadas``y_pred``

print(y_pred)





