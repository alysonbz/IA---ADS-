import matplotlib.pyplot as plt
from src.utils import load_grains_splited_datadet
from scipy.cluster.hierarchy import dendrogram, linkage

# Carregar dados corretamente
X_train, X_test, y_train, y_test = load_grains_splited_datadet()
X_dendro = X_train
varieties = y_train

print(f"Number of observations in X_dendro: {len(X_dendro)}")
print(f"Number of labels in varieties: {len(varieties)}")

if len(varieties) != len(X_dendro):
    raise ValueError("The number of labels must match the number of observations no dendrogram.")

# Calcular linkage
mergings = linkage(X_dendro, method='complete')

# Plotar o dendrograma
dendrogram(
    mergings,
    labels=varieties,
    leaf_rotation=90,
    leaf_font_size=10
)
plt.show()
