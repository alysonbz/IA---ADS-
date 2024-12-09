import pandas as pd

df = pd.read_csv('datasets/train.csv')

print("Valores nulos:")
print(df.isna().sum())

if df.isnull().values.any():
    df = df.dropna()

new_test = df.drop(['dual_sim', 'm_dep', 'blue', 'wifi', 'talk_time'], axis=1)

new_test['price_range'] = new_test['price_range'].fillna(new_test['price_range'].median())

print(new_test.isna().sum())

print(new_test.describe())

print(new_test['price_range'].value_counts())

new_test.to_csv('datasets/train_ajustado.csv', index=False)