import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = "../AV1/datasets/novo_Clean_Dataset.csv"
df = pd.read_csv(file_path)

df_encoded = pd.get_dummies(df, columns=['airline'], drop_first=True)

X = df_encoded.iloc[:, df_encoded.columns.str.startswith('airline')].values
y = df["price"].values

modelo = LinearRegression()
modelo.fit(X, y)

y_pred = modelo.predict(X)

print(f"Coeficientes: {modelo.coef_}")
print(f"Intercepto (b): {modelo.intercept_}")

RSS = np.sum((y - y_pred) ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

print(f"RSS: {RSS:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"RMSE: {RMSE:.2f}")
print(f"R²: {R_squared:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(range(len(y)), y, color='blue', alpha=0.6, label='Dados reais')
plt.plot(range(len(y)), y_pred, color='red', label='Predição')
plt.title(f"Regressão Linear com atributo 'airline' codificado")
plt.xlabel('Índice')
plt.ylabel('Preço')
plt.legend()
plt.show()
