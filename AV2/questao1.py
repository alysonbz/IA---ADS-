import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  # Adicionado LabelEncoder aqui
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Caminho local para o arquivo CSV
file_path = "flavors_of_cacao.csv"

# Carregar o dataset
df = pd.read_csv(file_path)

# Normalizar os nomes das colunas (remover espaços extras e converter para minúsculas)
df.columns = df.columns.str.strip().str.lower()

# Pré-processamento: Remover colunas irrelevantes
columns_to_drop = [
    'company\xa0\n(maker-if known)',
    'specific bean origin\nor bar name',
    'broad bean\norigin'
]
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=columns_to_drop, inplace=True)

# Tratar valores nulos (remover linhas com valores ausentes)
df.dropna(inplace=True)

# Converter a coluna 'cocoa percent' para numérica (remover o símbolo '%' e converter para float)
if 'cocoa\npercent' in df.columns:
    df['cocoa\npercent'] = df['cocoa\npercent'].str.replace('%', '', regex=True).astype(float)

# Criação de features
if 'cocoa\npercent' in df.columns:
    df['high_cocoa'] = (df['cocoa\npercent'] > 70).astype(int)  # Feature binária para alto teor de cacau
if 'review\ndate' in df.columns:
    df['decade'] = (df['review\ndate'] // 10) * 10  # Extrair a década da data de revisão

# Codificar variáveis categóricas
if 'company\nlocation' in df.columns:
    encoder = OneHotEncoder(sparse_output=False)
    location_encoded = encoder.fit_transform(df[['company\nlocation']])
    location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['company\nlocation']))
    df = pd.concat([df, location_df], axis=1)
    df.drop(columns=['company\nlocation'], inplace=True)

if 'bean\ntype' in df.columns:
    encoder = OneHotEncoder(sparse_output=False)
    bean_type_encoded = encoder.fit_transform(df[['bean\ntype']])
    bean_type_df = pd.DataFrame(bean_type_encoded, columns=encoder.get_feature_names_out(['bean\ntype']))
    df = pd.concat([df, bean_type_df], axis=1)
    df.drop(columns=['bean\ntype'], inplace=True)

# Discretizar o alvo ('rating') em classes
bins = [0, 2.5, 3.5, 5.0]  # Intervalos para as classes
labels = ['Baixo', 'Médio', 'Alto']
df['rating_class'] = pd.cut(df['rating'], bins=bins, labels=labels)

# Remover linhas com valores nulos no alvo
df = df.dropna(subset=['rating_class'])

# Filtrar classes com pelo menos 6 amostras
min_samples_per_class = 6
class_counts = df['rating_class'].value_counts()
valid_classes = class_counts[class_counts >= min_samples_per_class].index
df = df[df['rating_class'].isin(valid_classes)]

# Codificar o alvo como valores numéricos
label_encoder = LabelEncoder()  # Agora está definido corretamente
df['rating_class_encoded'] = label_encoder.fit_transform(df['rating_class'])

# Separar features e target
X = df.drop(columns=['rating', 'rating_class', 'rating_class_encoded'])  # Features
y = df['rating_class_encoded']  # Target codificado

# Preencher valores nulos nas features com a média
X = X.fillna(X.mean(numeric_only=True))

# Balancear as classes usando SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Definir o espaço de parâmetros para o GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),  # Testar valores de k de 1 a 20
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Métricas de distância
}

# Criar o modelo KNN
knn = KNeighborsClassifier()

# Configurar o GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Treinar o GridSearchCV
grid_search.fit(X_train, y_train)

# Melhores parâmetros e desempenho
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Avaliar no conjunto de teste
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Exibir os resultados
print("\n=== Resultados Finais ===")
print("Melhores parâmetros:", best_params)
print("Acurácia no conjunto de treino:", best_accuracy)
print("Acurácia no conjunto de teste:", test_accuracy)
print("Matriz de Confusão:\n", conf_matrix)
print("Relatório de Classificação:\n", class_report)
