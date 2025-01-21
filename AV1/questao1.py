#1
import pandas as pd
from sklearn.preprocessing import StandardScaler

classification = "Cancer_Data.csv"
regression = "possum.csv"

classification_df = pd.read_csv(classification)
regression_df = pd.read_csv(regression)

# Visualizar as primeiras linhas para verificar a estrutura do dataset
print("Preview do Dataset de Classificação:")
print(classification_df.head())

print("Preview do Dataset de Regressão:")
print(regression_df.head())

#2 Verificar se existem células vazias
print("\nDados ausentes no Dataset de Classificação:")
print(classification_df.isnull().sum())

print("\nDados ausentes no Dataset de Regressão:")
print(regression_df.isnull().sum())

# Remover colunas irrelevantes, 'Unnamed: 32' no dataset de classificação e 'case' no dataset de regressão
classification_df = classification_df.drop(columns=["Unnamed: 32"], errors="ignore")
regression_df = regression_df.drop(columns=["case"], errors="ignore")

# Excluir linhas com valores ausentes (ou pode optar por preencher com média ou mediana no caso de regressão)
classification_cleaned = classification_df.dropna()
regression_cleaned = regression_df.dropna()

print(f"\nNúmero de linhas removidas (classificação): {len(classification_df) - len(classification_cleaned)}")
print(f"Número de linhas removidas (regressão): {len(regression_df) - len(regression_cleaned)}")

# 4. Escolher colunas mais relevantes
classification_relevant_cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
regression_relevant_cols = ["site", "age", "hdlngth", "skullw", "totlngth", "taill"]


classification_final = classification_cleaned[classification_relevant_cols]
regression_final = regression_cleaned[regression_relevant_cols]

# Renomear as classes no dataset de classificação (M para 1, B para 0)
classification_final.loc[:, "diagnosis"] = classification_final["diagnosis"].map({"M": 1, "B": 0})

# Normalização (aplicar somente nas colunas numéricas)
scaler = StandardScaler()
regression_final.loc[:, ["age", "hdlngth", "skullw", "totlngth", "taill"]] = scaler.fit_transform(
    regression_final[["age", "hdlngth", "skullw", "totlngth", "taill"]]
)

# 6. Necessidade de mais pré-processamento???
print("\nDataset de Classificação Processado:")
print(classification_final.head())

# 7. Salvar os datasets atualizados
classification_final.to_csv("Cancer_Data_Adjusted.csv", index=False)
regression_final.to_csv("Possum_Data_Adjusted.csv", index=False)

print("\nDatasets processados e salvos com sucesso.")
