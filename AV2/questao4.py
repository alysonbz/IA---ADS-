import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


def load_adjusted_dataset(path="dataset/star_classification_ajustado.csv"):
    """Carrega o dataset ajustado, que deve conter a coluna 'class'"""
    df = pd.read_csv(path)
    return df


def select_two_attributes_lasso(df, alpha=0.01):
    """Seleciona dois atributos mais relevantes usando Lasso"""
    # Garante que 'class' existe e é numérica (0=GALAXY, 1=STAR, 2=QSO)
    if 'class' not in df.columns:
        raise ValueError("Coluna 'class' não encontrada no DataFrame.")

    # y = class, X = todas as colunas exceto 'class'
    y = df['class']
    X = df.drop('class', axis=1)

    # Normaliza X para Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplica Lasso
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_scaled, y)

    # Extrair coeficientes e ordenar por magnitude
    coef_abs = np.abs(lasso.coef_)
    feature_names = X.columns.tolist()
    sorted_features = sorted(zip(feature_names, coef_abs), key=lambda x: x[1], reverse=True)

    # Selecionar as duas colunas de maior relevância
    top2 = [f[0] for f in sorted_features[:2]]
    print("Atributos selecionados pelo Lasso (top 2):", top2)
    return top2


def find_best_k_silhouette(X, k_min=2, k_max=10):
    """Retorna o melhor k (número de clusters) baseado no índice de silhueta"""
    results = []
    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append((k, sil))
        if sil > best_score:
            best_score = sil
            best_k = k

    return best_k, results


def plot_silhouette_results(sil_results, folder="resultados/questao4"):
    """Plota e salva o gráfico de silhueta para diferentes k"""
    os.makedirs(folder, exist_ok=True)

    df_sil = pd.DataFrame(sil_results, columns=["k", "silhouette"])
    df_sil.to_csv(f"{folder}/silhouette_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df_sil["k"], df_sil["silhouette"], marker='s', linestyle='-')
    plt.title("Índice de Silhueta vs Número de Clusters (Questão 4)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Coeficiente de Silhueta (médio)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/silhouette_plot.png")
    plt.close()


def main():
    question_folder = "resultados/questao4"
    os.makedirs(question_folder, exist_ok=True)

    # Carregar dataset
    df = load_adjusted_dataset()
    print("Dimensões do dataset ajustado:", df.shape)

    # Selecionar 2 atributos mais relevantes via Lasso
    top2_features = select_two_attributes_lasso(df, alpha=0.01)

    # Manter apenas as top2 features para cluster
    y_class = df["class"].copy()
    # Filtrar apenas as colunas do top2
    df2 = df[top2_features].copy()

    # Normalizar as colunas selecionadas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df2)

    # Achar melhor k pelo índice de silhueta
    best_k, sil_results = find_best_k_silhouette(X_scaled, k_min=2, k_max=10)
    print(f"Melhor k segundo silhueta: {best_k}")

    plot_silhouette_results(sil_results, folder=question_folder)

    # Executar KMeans com best_k
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Crosstab
    df_crosstab = pd.crosstab(index=y_class, columns=labels)
    print("\n=== Crosstab: Classes (linhas) vs Clusters (colunas) ===")
    print(df_crosstab)

    # Salvar crosstab em CSV
    df_crosstab.to_csv(f"{question_folder}/crosstab_clusters_vs_class.csv")
    print(f"Crosstab salvo em: {question_folder}/crosstab_clusters_vs_class.csv")


if __name__ == "__main__":
    main()
