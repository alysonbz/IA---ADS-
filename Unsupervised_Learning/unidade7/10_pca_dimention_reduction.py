from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset

# Load dataset and drop 'specie' column
samples = load_fish_dataset()
samples = samples.drop(['specie'], axis=1)

# Standardize the data
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

# Create a PCA model with an adequate number of components
pca = PCA(n_components=2)  # Reduzindo para 2 dimensões para visualização

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

# Visualize scatter plot with dimension reduced
import matplotlib.pyplot as plt

plt.scatter(pca_features[:, 0], pca_features[:, 1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization of Fish Dataset')
plt.show()
