import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


# 1. Importar o dataset
df = pd.read_csv(r'C:\Users\pinheiroiwnl\Desktop\AV1\IA---ADS-\AV1\diabetes.csv')

# 2. Verificar células vazias ou NaN
print(df.isnull().sum())

# 3. Identificar as colunas mais relevantes

# Create X and y arrays
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"].values
sales_columns = X.columns

# Instanciar um modelo de regressão Lasso
lasso = Lasso(alpha=0.3)

# Calcular e imprimir os coeficientes
lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()

# 4. Mostrar o dataframe final e a distribuição de classes
print("Dataframe final:")
print(df.head())

# Plotar gráfico das distribuições de classes
class_distribution = df["Outcome"].value_counts()
class_distribution.plot(kind="bar", figsize=(8, 6), color=["skyblue", "salmon"], rot=0)
plt.title("Distribuição de Classes (Outcome)")
plt.xlabel("Classes")
plt.ylabel("Contagem")
plt.xticks(ticks=[0, 1], labels=["Não", "Sim"])
plt.show()


# 5. Converter coluna de classes para atributos numéricos (se necessário)
# No dataset, 'Outcome' já está como valores numéricos, então não é necessária conversão.

# 6. Avaliar necessidade de mais pré-processamento
# Para este exercício inicial, o dataset parece adequado para modelagem. .

# 7. Salvar o dataset atualizado
df.to_csv('diabestes_ajustado.csv', index=True)