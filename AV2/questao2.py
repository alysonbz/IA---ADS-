import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_adjusted_dataset(path="dataset/star_classification_ajustado.csv"):
    """Carrega o dataset ajustado"""
    df = pd.read_csv(path)
    return df


def prepare_features(df):
    """Remove a coluna alvo 'class' e faz normalização"""
    if 'class' in df.columns:
        df = df.drop('class', axis=1)

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled


def elbow_method(X, k_max=10):
    """ COTOVELO: Aplica KMeans variando k de 2 até k_max"""
    elbow_results = []
    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        elbow_results.append((k, kmeans.inertia_))
    return elbow_results


def silhouette_analysis(X, k_max=10):
    """ SILHUETA: Aplica KMeans variando k de 2 até k_max"""
    silhouette_results = []
    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil_score = silhouette_score(X, labels)
        silhouette_results.append((k, sil_score))
    return silhouette_results


def save_results(elbow_results, silhouette_results, folder="resultados/questao2"):
    """Salva os resultados de Elbow e Silhouette em arquivos CSV e gráficos"""
    os.makedirs(folder, exist_ok=True)

    # Salvar CSV
    elbow_df = pd.DataFrame(elbow_results, columns=["k", "inertia"])
    elbow_df.to_csv(f"{folder}/elbow_results.csv", index=False)

    silhouette_df = pd.DataFrame(silhouette_results, columns=["k", "silhouette"])
    silhouette_df.to_csv(f"{folder}/silhouette_results.csv", index=False)

    # Plot do método do cotovelo (Elbow)
    plt.figure(figsize=(8, 5))
    plt.plot(elbow_df["k"], elbow_df["inertia"], marker='o', linestyle='-')
    plt.title("Método do Cotovelo (Elbow)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia (Inertia)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/elbow_plot.png")
    plt.close()

    # Plot do método da silhueta
    plt.figure(figsize=(8, 5))
    plt.plot(silhouette_df["k"], silhouette_df["silhouette"], marker='s', linestyle='-')
    plt.title("Análise de Silhueta")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Coeficiente de Silhueta (médio)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/silhouette_plot.png")
    plt.close()


def main():
    # Define a pasta para armazenar resultados
    question_folder = "resultados/questao2"
    os.makedirs(question_folder, exist_ok=True)

    # Carregar Dataset Ajustado
    df = load_adjusted_dataset("dataset/star_classification_ajustado.csv")
    print(f"Tamanho do dataset carregado: {df.shape}")

    # Preparar features (remover a coluna 'class' e normalizar)
    X = prepare_features(df)

    # Executar Elbow e Silhouette
    elbow_results = elbow_method(X, k_max=10)  # Ajuste k_max se necessário
    silhouette_results = silhouette_analysis(X, k_max=10)

    # Salvar e plotar resultados
    save_results(elbow_results, silhouette_results, folder=question_folder)

    print("Resultados salvos com sucesso!")
    print(f"Arquivos salvos em {question_folder}")


if __name__ == "__main__":
    main()
