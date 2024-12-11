import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

file_path = "../AV1/datasets/novo_Clean_Dataset.csv"
df = pd.read_csv(file_path)

def RS(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def RSS(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def RMSE(y_true, y_pred):
    mse = MSE(y_true, y_pred)
    return np.sqrt(mse)

def k_fold_cross_validation(X, y, k=5):
    fold_size = len(X) // k
    rss_list = []
    mse_list = []
    rmse_list = []
    r2_list = []

    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i != k - 1 else len(X)
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
        y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rss = RSS(y_val, y_pred)
        mse = MSE(y_val, y_pred)
        rmse = RMSE(y_val, y_pred)
        r2 = RS(y_val, y_pred)

        rss_list.append(rss)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)

    avg_rss = np.mean(rss_list)
    avg_mse = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_r2 = np.mean(r2_list)

    return avg_rss, avg_mse, avg_rmse, avg_r2

label_encoder = LabelEncoder()
df["class"] = label_encoder.fit_transform(df["class"])

X = df["class"].values.reshape(-1, 1)
y = df["price"].values.reshape(-1, 1)
scaler_X = MinMaxScaler()
X_normalizado = scaler_X.fit_transform(X)

avg_rss, avg_mse, avg_rmse, avg_r2 = k_fold_cross_validation(X_normalizado, y, k=5)

print(f"RSS médio: {avg_rss:.2f}")
print(f"MSE médio: {avg_mse:.2f}")
print(f"RMSE médio: {avg_rmse:.2f}")
print(f"R² médio: {avg_r2:.2f}")
