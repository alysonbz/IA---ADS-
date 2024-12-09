import pandas as pd
import matplotlib.pyplot as plt


def load_dataset_stars():
    df = pd.read_csv("dataset/classification/star_classification.csv")
    return df


stars = load_dataset_stars()

# Verificar se há valores nulos
print(50 * "-" + " NULLS " + 50 * "-")
print(stars.isna().sum())

# header do dataframe
print(50 * "-" + " HEADER " + 50 * "-")
print(stars.head())

# info do dataframe
print(50 * "-" + " INFO " + 50 * "-")
print(stars.info())

# descricao do dataframe
print(50 * "-" + " DESCRIPTION " + 50 * "-")
print(stars.describe())

# selecionar colunas relevantes
relevant_columns = ['u', 'g', 'r', 'i', 'z', 'redshift', 'class']
stars_relevant = stars[relevant_columns].copy()

# analisar distribuição das classes
print(stars_relevant['class'].value_counts())

class_mapping = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
stars_relevant['class'] = stars_relevant['class'].map(class_mapping)

# salvar o dataset atualizado
stars_relevant.to_csv("dataset/classification/star_classification_ajustado.csv", index=False)

# gráfico de distribuição de classes
class_counts = stars_relevant['class'].value_counts()
class_labels = ['GALAXY', 'STAR', 'QSO']

plt.bar(class_labels, class_counts, color=['blue', 'orange', 'green'])
plt.xlabel('Classes')
plt.ylabel('Frequência')
plt.title('Distribuição de Classes no Dataset')
plt.show()
