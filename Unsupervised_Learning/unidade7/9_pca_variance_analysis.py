# Perform the necessary imports
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import load_fish_dataset

samples = load_fish_dataset()
samples = samples.drop(['specie'],axis=1)


# Create scaler: scaler
scaler = ___

# Create a PCA instance: pca
pca = ___

# Create pipeline: pipeline
pipeline = ___(__,__)

# Fit the pipeline to 'samples'
___

# Plot the explained variances
features = ____
plt.bar(____, ____)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
