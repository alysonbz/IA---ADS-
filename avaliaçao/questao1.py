import pandas as pd

# Carregando o arquivo CSV
file_path = r"C:\Users\Davi Gomes\Documents\Faculdade\IA\avaliaçao\datasets\healthcare-dataset-stroke-data-classification.csv"
data = pd.read_csv(file_path)

print(data.head())
print(data.info())

print(data.isnull().sum())

data_cleaned = data.dropna() #excluindo os na
print(data_cleaned.info())

#selecionando as colunas mais relevantes
relevant_columns = ['gender', 'age', 'hypertension', 'heart_disease',
                    'ever_married', 'work_type', 'Residence_type',
                    'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
data_relevant = data_cleaned[relevant_columns]

print(data_relevant.head())  #colunas selecionadas

class_distribution = data_relevant['stroke'].value_counts()
print("Distribuição de classes:\n", class_distribution)

# Aplicando one-hot encoding para transformar colunas categóricas
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data_final = pd.get_dummies(data_relevant, columns=categorical_columns, drop_first=True)

print(data_final.info())  #novo formato dos dados

#dataset final
output_path = 'healtcare_ajustado.csv'
data_final.to_csv(output_path, index=False)
print(f"Dataset ajustado salvo em: {output_path}")


