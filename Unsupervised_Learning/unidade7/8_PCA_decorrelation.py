# Import PCA
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from src.utils import load_grains_dataset


grains = load_grains_dataset()
grains = grains.drop(['variety', 'variety_number'], axis=1)
# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:, 0]

# Assign 1st column of pca_features: ys
ys = pca_features[:, 1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)
