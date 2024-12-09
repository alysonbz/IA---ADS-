import pandas as pd
import os

# Caminho do arquivo de entrada
file_path = os.path.join(os.path.dirname(__file__), 'drug200.csv')

# Carregar o dataset
dataset = pd.read_csv(file_path)

# Pré-processamento
dataset_cleaned = dataset.dropna()
if 'Drug' in dataset_cleaned.columns:
    dataset_cleaned['Drug'] = dataset_cleaned['Drug'].astype('category').cat.codes

# Novo caminho de saída
output_file_path = os.path.join(os.path.dirname(__file__), 'drug200_ajustado.csv')

# Salvar o arquivo ajustado
dataset_cleaned.to_csv(output_file_path, index=False)
print(f"Arquivo salvo com sucesso em: {output_file_path}")