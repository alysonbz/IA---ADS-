import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_adjusted_dataset(path="dataset/star_classification_ajustado.csv"):
    """Carrega o dataset ajustado"""
    df = pd.read_csv(path)
    return df


def select_two_attributes_lasso(df, alpha=0.01):
    """Utiliza o método de Lasso para identificar os dois atributos mais relevantes"""
    # Garante que 'class' existe e é numérica (GALAXY=0, STAR=1, QSO=2)
    if 'class' not in df.columns:
        raise ValueError("Coluna 'class' não encontrada no DataFrame.")

    # Definir y como a coluna 'class'
    y = df['class']
    # X são todas as outras colunas (exceto 'class')
    X = df.drop('class', axis=1)

    # Normaliza X para o Lasso
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


def elbow_method(X, k_max=10):
    """Aplica KMeans variando k de 2 até k_max e retorna lista de (k, inercia)"""
    elbow_results = []
    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        elbow_results.append((k, kmeans.inertia_))
    return elbow_results


def silhouette_analysis(X, k_max=10):
    """Aplica KMeans variando k de 2 até k_max e retorna lista de (k, silhueta)"""
    silhouette_results = []
    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil_score = silhouette_score(X, labels)
        silhouette_results.append((k, sil_score))
    return silhouette_results


def save_and_plot_elbow_silhouette(elbow_results, silhouette_results, folder="resultados/questao3"):
    """Salva em CSV e plota os gráficos do método do cotovelo e do índice de silhueta"""
    os.makedirs(folder, exist_ok=True)

    # Salvar CSV
    elbow_df = pd.DataFrame(elbow_results, columns=["k", "inertia"])
    elbow_df.to_csv(f"{folder}/elbow_results.csv", index=False)

    silhouette_df = pd.DataFrame(silhouette_results, columns=["k", "silhouette"])
    silhouette_df.to_csv(f"{folder}/silhouette_results.csv", index=False)

    # Plot Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(elbow_df["k"], elbow_df["inertia"], marker='o', linestyle='-')
    plt.title("Método do Cotovelo (Elbow) - 2 atributos (Lasso)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia (Inertia)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/elbow_plot.png")
    plt.close()

    # Plot Silhouette
    plt.figure(figsize=(8, 5))
    plt.plot(silhouette_df["k"], silhouette_df["silhouette"], marker='s', linestyle='-')
    plt.title("Análise de Silhueta - 2 atributos (Lasso)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Coeficiente de Silhueta (médio)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/silhouette_plot.png")
    plt.close()


def scatter_clusters(X, k, folder, suffix="", random_state=42):
    """Gera um scatterplot dos clusters para análise visual usando os dois atributos"""
    os.makedirs(folder, exist_ok=True)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.title(f"Scatterplot - k={k}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(f"{folder}/scatter_k_{k}{suffix}.png")
    plt.close()


def main():
    # Pasta de resultados
    question_folder = "resultados/questao3"
    os.makedirs(question_folder, exist_ok=True)

    # Carregar dataset ajustado (com 'class')
    df = load_adjusted_dataset("dataset/star_classification_ajustado.csv")
    print("Dimensões do dataset ajustado:", df.shape)

    # Selecionar 2 atributos mais relevantes usando Lasso (levando em conta a coluna class)
    top2_features = select_two_attributes_lasso(df, alpha=0.01)

    # Descartar 'class'
    df2 = df[top2_features].copy()

    # Normalizar esses dois atributos
    scaler = StandardScaler()
    X_2attr = scaler.fit_transform(df2)

    # Aplicar Elbow e Silhouette
    elbow_res = elbow_method(X_2attr, k_max=10)
    sil_res = silhouette_analysis(X_2attr, k_max=10)

    # Salvar e plotar
    save_and_plot_elbow_silhouette(elbow_res, sil_res, folder=question_folder)

    # Identificar melhor k para Elbow e Silhouette
    elbow_k = 3  # Ajuste manual se precisar (baseado em elbow_plot.png)
    best_k_sil = max(sil_res, key=lambda x: x[1])[0]

    print(f"K escolhido pelo Elbow: {elbow_k}")
    print(f"K escolhido pela Silhueta: {best_k_sil}")

    # Se forem diferentes, gerar scatterplots de ambos
    if elbow_k != best_k_sil:
        scatter_clusters(X_2attr, elbow_k, folder=question_folder, suffix="_elbow")
        scatter_clusters(X_2attr, best_k_sil, folder=question_folder, suffix="_silhouette")
        print("Os valores de k foram diferentes, scatterplots salvos para comparação.")
    else:
        # Caso sejam iguais, só gera 1 scatterplot
        scatter_clusters(X_2attr, elbow_k, folder=question_folder)
        print("Os valores de k foram iguais, scatterplot salvo.")


if __name__ == "__main__":
    main()
