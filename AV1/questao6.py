import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Carregar o dataset
data = pd.read_csv(r'C:\Users\pinheiroiwnl\Desktop\AV1\IA---ADS-\AV1\king_county_preprocessed.csv')

# 2. Definir o alvo para regressão
# O alvo é o preço da casa ('price')
target = "price"

# 3. Identificar a correlação das variáveis com o alvo
correlations = data.corr()[target].sort_values(ascending=False)
print("Correlação das variáveis com o preço:")
print(correlations)

# 4. Selecionar o atributo mais relevante
most_relevant_feature = correlations.index[1]

# 5. Implementar a regressão linear com o atributo mais relevante
X = data[[most_relevant_feature]]
y = data[target]

model = LinearRegression()
model.fit(X, y)

# Previsões
y_pred = model.predict(X)

# 6. Calcular métricas de avaliação
RSS = np.sum((y - y_pred) ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

print(f"RSS: {RSS:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"RMSE: {RMSE:.2f}")
print(f"R^2: {R_squared:.4f}")

# 7. Visualizar a reta de regressão
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[most_relevant_feature], y=y, label="Dados reais", alpha=0.6)
plt.plot(X[most_relevant_feature], y_pred, color="red", label="Reta de Regressão")
plt.title(f"Regressão Linear: {most_relevant_feature} vs {target}")
plt.xlabel(most_relevant_feature)
plt.ylabel(target)
plt.legend()
plt.show()