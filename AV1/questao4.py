from src.utils import load_new_dataframe_gender_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

scaler = StandardScaler()
gender_df = load_new_dataframe_gender_classification()

X = gender_df.drop(["gender"], axis=1).values
y = gender_df["gender"].values

X_norm = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.25, random_state=42, stratify=y)

neighbors = np.arange(1, 15)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("acuracy on train: ",train_accuracies, '\n',"acuracy on test: ", test_accuracies)

plt.title("Plot da Accuracies do KNN")

plt.plot(neighbors, train_accuracies.values(), label="Training Accuracies")

plt.plot(neighbors, test_accuracies.values(), label="Test Accuracies")

plt.legend()
plt.xlabel("Numero de Vizinhos")
plt.ylabel("Accuracies")

plt.show()