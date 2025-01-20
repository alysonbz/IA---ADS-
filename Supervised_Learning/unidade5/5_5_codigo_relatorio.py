from src.utils import load_diabetes_clean_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
import seaborn as sns
import matplotlib.pyplot as plt


diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

print("Desempenho do KNN:")
print(classification_report(y_test, y_pred_knn))

print("Desempenho da Regressão Logística:")
print(classification_report(y_test, y_pred_logreg))

# Matriz de confusão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title("Matriz de Confusão - KNN")

sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title("Matriz de Confusão - Regressão Logística")
plt.show()

# Curva ROC
RocCurveDisplay.from_estimator(knn, X_test, y_test, name="KNN")
RocCurveDisplay.from_estimator(logreg, X_test, y_test, name="Regressão Logística")
plt.title("Curvas ROC")
plt.show()

results = pd.DataFrame({
    "Modelo": ["KNN", "Regressão Logística"],
    "Acurácia": [accuracy_knn, accuracy_logreg]
})

print("Resumo dos Resultados:")
print(results)
