import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
file_path = "../AV1/datasets/novo_Clean_Dataset.csv"
df = pd.read_csv(file_path)

label_encoder = LabelEncoder()
df["class"] = label_encoder.fit_transform(df["class"])

X = df["class"].values.reshape(-1, 1)
y = df["price"].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_normalizado = scaler_X.fit_transform(X)
y_normalizado = scaler_y.fit_transform(y)

modelo = LinearRegression()
modelo.fit(X_normalizado, y_normalizado)

y_pred_normalizado = modelo.predict(X_normalizado)

y_pred = scaler_y.inverse_transform(y_pred_normalizado)

coef_angular = modelo.coef_[0][0]
intercepto = modelo.intercept_[0]

RSS = np.sum((y.flatten() - y_pred.flatten()) ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

print(f"Coeficiente angular (m): {coef_angular:.4f}")
print(f"Intercepto (b): {intercepto:.4f}")
print(f"RSS: {RSS / 1e9:.2f} bilhões")
print(f"MSE: {MSE:.2f}")
print(f"RMSE: {RMSE:.2f}")
print(f"R²: {R_squared:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Dados reais')
plt.plot(X, y_pred, color='red', label='Reta de regressão')
plt.title("Regressão Linear: Class vs Price")
plt.xlabel("Class")
plt.ylabel("Price")
plt.legend()
plt.show()
