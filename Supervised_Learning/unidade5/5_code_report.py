import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from src.utils import load_diabetes_clean_dataset


def plot_confusion_matrix(cm, title, labels=['Non-Diabetic', 'Diabetic']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
    plt.close()


diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

#normalização
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42, stratify=y)
#modelos utilizados
knn = KNeighborsClassifier(n_neighbors=7)
logreg = LogisticRegression(random_state=42)

#treinamento e resultados de ambos modelos
results = {}
for name, model in [('KNN', knn), ('Logistic Regression', logreg)]:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    conf_matrix = confusion_matrix(y_test, y_pred)
    results[name] = {
        'confusion_matrix': conf_matrix,
        'classification_report': classification_report(y_test, y_pred),
        'auc_score': roc_auc_score(y_test, y_pred_proba),
        'fpr': None,
        'tpr': None
    }

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    results[name]['fpr'] = fpr
    results[name]['tpr'] = tpr
    # matriz de confusão
    plot_confusion_matrix(conf_matrix, name)
    print(f"\n{'-' * 50}")
    print(f"{name} Results:")
    print(f"{'-' * 50}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(results[name]['classification_report'])
    print(f"AUC Score: {results[name]['auc_score']:.4f}")

# Plot ROC
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Random')

colors = {'KNN': 'blue', 'Logistic Regression': 'red'}
for name in results:
    plt.plot(
    results[name]['fpr'],
        results[name]['tpr'],
        color=colors[name],
        label=f'{name} (AUC = {results[name]["auc_score"]:.3f})'
    )

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png')
plt.close()