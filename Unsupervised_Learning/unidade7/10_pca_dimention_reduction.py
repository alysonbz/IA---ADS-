
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder

samples = load_fish_dataset()
samples = samples.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)


# Create a PCA model with components in adequate number: pca
pca = __

# Fit the PCA instance to the scaled samples
__

# Transform the scaled samples: pca_features
__

# Print the shape of pca_features
__

#vizualize scatter plot with dimension reduced
__