import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression


class KFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, y_true, y_pred):
        return np.sum(np.square(y_pred - np.mean(y_true))) / np.sum(np.square(y_true - np.mean(y_true)))

    def cross_val_score(self, model, X, y):
        indices = np.random.permutation(len(y))
        fold_size = len(y) // self.n_splits

        scores = []
        for i in range(self.n_splits):
            test_idx = indices[i * fold_size:(i + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            scores.append(self._compute_score(y_test, model.predict(X_test)))

        return scores


# Carregar dados
df = load_sales_clean_dataset()
X, y = df["tv"].values.reshape(-1, 1), df["sales"].values

# Criar e executar validação cruzada
cv_scores = KFold(n_splits=6).cross_val_score(LinearRegression(), X, y)

# Exibir resultados
print(cv_scores)
print(f"Média: {np.mean(cv_scores):.4f}")
print(f"Desvio Padrão: {np.std(cv_scores):.4f}")
