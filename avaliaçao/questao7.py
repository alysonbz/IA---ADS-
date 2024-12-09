import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


#calcular RSS
def calculate_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

#calcular MSE
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#calcular RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

#calcular R²
def calculate_r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

data = pd.read_csv('Lenovo_ajustado.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#armazenar resultados
rss_list = []
mse_list = []
rmse_list = []
r2_list = []

# Implementação manual do k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rss = calculate_rss(y_test, y_pred)
    mse = calculate_mse(y_test, y_pred)
    rmse = calculate_rmse(y_test, y_pred)
    r_squared = calculate_r_squared(y_test, y_pred)

    rss_list.append(rss)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r_squared)

mean_rss = np.mean(rss_list)
mean_mse = np.mean(mse_list)
mean_rmse = np.mean(rmse_list)
mean_r2 = np.mean(r2_list)

print(f"RSS Médio: {mean_rss:.2f}")
print(f"MSE Médio: {mean_mse:.2f}")
print(f"RMSE Médio: {mean_rmse:.2f}")
print(f"R² Médio: {mean_r2:.2f}")

def plot_metrics(metrics, metric_name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics) + 1), metrics, marker='o', linestyle='--')
    plt.title(f'{metric_name} para cada fold')
    plt.xlabel('Fold')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.show()

# Plotar as métricas
plot_metrics(rss_list, 'RSS')
plot_metrics(mse_list, 'MSE')
plot_metrics(rmse_list, 'RMSE')
plot_metrics(r2_list, 'R²')