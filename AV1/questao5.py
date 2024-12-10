import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "../AV1/datasets/Clean_Dataset.csv"
df = pd.read_csv(file_path)

df = df.dropna()
df = df.drop(df.columns[[0, 2]], axis=1)

colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns

df_one_hot = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

matriz_correlacao = df_one_hot.corr()


plt.figure(figsize=(100, 80))
sns.heatmap(matriz_correlacao, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação (Com Variáveis Codificadas)")
plt.show()
