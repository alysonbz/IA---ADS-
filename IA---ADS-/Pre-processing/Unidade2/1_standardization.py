from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

wine = load_wine_dataset()

X = wine.drop(['Quality'],axis=1)

y = wine['Quality'].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test = ____(____, ____, stratify=____, random_state=42)

knn = KNeighborsClassifier()

# Aplique a função fit do knn
knn.____(____, ____)

# mostre o acerto do algoritmo
print(knn.____(____))