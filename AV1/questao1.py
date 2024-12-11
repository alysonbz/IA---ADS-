import pandas as pd

file_path = "../AV1/datasets/creditcard_2023.csv"
df = pd.read_csv(file_path)

df = df.dropna()
df = df.drop(["id"], axis=1)
print(df)
print(df['Class'].value_counts())

dataframe_ajustado = "../AV1/datasets/novo_creditcard.csv"
df.to_csv(dataframe_ajustado, index=False)
