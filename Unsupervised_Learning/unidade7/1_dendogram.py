import matplotlib.pyplot as plt
from src.utils import load_grains_splited_datadet
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
X_train, samples, y_train, varieties = load_grains_splited_datadet()

# Calculate the linkage: mergings
mergings = linkage(X_train, method='complete')

if len(varieties) != len(X_train):
    raise ValueError("The number of labels must match the number of observations in the dataset.")


# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=10)
plt.show()
