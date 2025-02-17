import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_dataset_stars(path="dataset/star_classification.csv"):
    """Carrega o dataset de classificação de estrelas"""
    df = pd.read_csv(path)
    return df


def prepare_dataset(df):
    """Prepara o dataset para análise"""
    relevant_columns = ['u', 'g', 'r', 'i', 'z', 'redshift', 'class']
    stars_relevant = df[relevant_columns].copy()

    class_mapping = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
    stars_relevant['class'] = stars_relevant['class'].map(class_mapping)

    os.makedirs("dataset", exist_ok=True)
    stars_relevant.to_csv("dataset/star_classification_ajustado.csv", index=False)

    return stars_relevant


def normalize_features(X):
    """Aplica normalização logarítmica nas features, exceto 'redshift'"""
    cols_to_normalize = [col for col in X.columns if col != 'redshift']
    min_values = X[cols_to_normalize].min()
    X_normalized = X.copy()
    X_normalized[cols_to_normalize] = np.log1p(X[cols_to_normalize] - min_values + 1)
    return X_normalized


def run_grid_search_knn(X_train, y_train):
    """Executa GridSearchCV para encontrar os melhores parâmetros para KNN"""
    # Matriz de covariância inversa para Mahalanobis
    cov_matrix = np.cov(X_train, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)  # Evita erros se cov_matrix não for invertível

    param_grid = [
        {'n_neighbors': list(range(1, 21)), 'metric': ['chebyshev']},
        {'n_neighbors': list(range(1, 21)), 'metric': ['manhattan']},
        {'n_neighbors': list(range(1, 21)), 'metric': ['euclidean']},
        {
            # Pseudo-inversa da matriz de covariância
            'n_neighbors': list(range(1, 21)),
            'metric': ['mahalanobis'],
            'metric_params': [{'VI': inv_cov_matrix}]
        }
    ]

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', error_score='raise')
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model, X_test, y_test):
    """Avalia o modelo no conjunto de teste e exibe o classification report"""
    y_pred = model.predict(X_test)
    print("\n=== Avaliação no Conjunto de Teste ===")
    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    return y_pred


def generate_confusion_matrix(model, X_test, y_test, folder):
    """ Gera e salva a matriz de confusão do melhor modelo"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot e salva em png
    disp.plot(cmap='Blues', values_format='d')
    os.makedirs(folder, exist_ok=True)
    plt.title("Matriz de Confusão - Melhor Modelo")
    plt.tight_layout()
    plt.savefig(f"{folder}/confusion_matrix.png")
    plt.close()


def save_final_report(grid_search, model, X_test, y_test, folder):
    """Salva em um arquivo .txt as informações finais"""
    os.makedirs(folder, exist_ok=True)

    # Resultados finais no conjunto de teste
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    clf_report = classification_report(y_test, y_pred)

    with open(f"{folder}/final_results.txt", "w") as f:
        f.write("===== FINAL REPORT =====\n")
        f.write(f"Melhores parâmetros (GridSearchCV): {grid_search.best_params_}\n")
        f.write(f"Acurácia (validação cruzada): {grid_search.best_score_ * 100:.2f}%\n\n")
        f.write(f"Acurácia (teste): {test_accuracy:.2f}%\n\n")
        f.write("=== Classification Report ===\n")
        f.write(clf_report)


def save_metric_tables(metric_scores, folder):
    """Salva tabelas CSV com valores de (K, mean_score) para cada distância"""
    os.makedirs(folder, exist_ok=True)
    for metric, scores in metric_scores.items():
        if scores:
            scores.sort()
            df = pd.DataFrame(scores, columns=["K", "mean_score"])
            df.to_csv(f"{folder}/{metric}_scores.csv", index=False)


def plot_individual_metrics(metric_scores, folder):
    """Plota cada distância individualmente em um gráfico"""
    os.makedirs(folder, exist_ok=True)
    markers = {'euclidean': 'o', 'manhattan': 's', 'chebyshev': '^', 'mahalanobis': 'D'}

    for metric, scores in metric_scores.items():
        if scores:
            scores.sort()
            k_vals = [x[0] for x in scores]
            mean_vals = [x[1] * 100 for x in scores]

            plt.figure(figsize=(8, 5))
            plt.plot(k_vals, mean_vals, marker=markers[metric], linestyle='-',
                     label=f"{metric.capitalize()} Distance")
            plt.title(f"Acurácia do KNN - {metric.capitalize()}")
            plt.xlabel("Valor de K")
            plt.ylabel("Acurácia (%)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{folder}/{metric}_plot.png")
            plt.close()


def plot_grid_search_results(grid_search, folder):
    """Plota e salva os resultados do GridSearchCV em um gráfico combinado e gera os csv's"""
    os.makedirs(folder, exist_ok=True)

    results = grid_search.cv_results_
    mean_scores = results['mean_test_score']
    params = results['params']

    # Dicionário para agrupar pontuações de cada métrica
    metric_scores = {metric: [] for metric in ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']}

    # Preencher metric_scores
    for i, param in enumerate(params):
        metric_used = param['metric']
        if metric_used in metric_scores:
            metric_scores[metric_used].append((param['n_neighbors'], mean_scores[i]))

    # Salvar tabelas CSV de cada métrica
    save_metric_tables(metric_scores, folder)

    # Plotar gráficos individuais por métrica
    plot_individual_metrics(metric_scores, folder)

    # Plot combinado
    plt.figure(figsize=(12, 7))
    markers = {'euclidean': 'o', 'manhattan': 's', 'chebyshev': '^', 'mahalanobis': 'D'}

    for metric, scores in metric_scores.items():
        if scores:
            scores.sort()
            plt.plot([x[0] for x in scores], [x[1] * 100 for x in scores],
                     marker=markers[metric], linestyle='-', label=metric.capitalize())

    plt.title('Acurácia do KNN para Diferentes Valores de K e Métricas')
    plt.xlabel('Valor de K')
    plt.ylabel('Acurácia (%)')
    plt.grid(True)
    plt.legend()

    best_k = grid_search.best_params_['n_neighbors']
    best_metric = grid_search.best_params_['metric']
    plt.axvline(best_k, color='red', linestyle='--')
    plt.text(best_k + 0.5, plt.ylim()[0] + 5,
             f'Melhor K={best_k}\nMétrica={best_metric}',
             color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{folder}/knn_gridsearch_results.png")
    plt.show()


def main():
    # Define qual é a pasta da questão
    question_folder = "resultados/questao1"

    # Carregar e preparar o dataset
    stars_original = load_dataset_stars()
    stars_prepared = prepare_dataset(stars_original)
    stars = pd.read_csv('dataset/star_classification_ajustado.csv')

    # Separar features e target
    X = stars.drop(["class"], axis=1)
    y = stars['class']

    # Normalizar as features
    X_normalized = normalize_features(X)

    # Dividir o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )

    # Executar GridSearchCV para encontrar os melhores parâmetros
    grid_search = run_grid_search_knn(X_train, y_train)

    print("\n=== Resultados do GridSearchCV ===")
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor acurácia na validação cruzada: {grid_search.best_score_ * 100:.2f}%")

    best_knn = grid_search.best_estimator_
    evaluate_model(best_knn, X_test, y_test)

    # Gera e salva a matriz de confusão
    generate_confusion_matrix(best_knn, X_test, y_test, question_folder)

    # Salvar relatório final
    save_final_report(grid_search, best_knn, X_test, y_test, folder=question_folder)

    # Gera tabelas CSV e plots
    plot_grid_search_results(grid_search, folder=question_folder)


if __name__ == "__main__":
    main()
