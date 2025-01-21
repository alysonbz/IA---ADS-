import pandas as pd

# Perform the necessary imports
from ____ import ____
from ____ import ____
from ____ import ____

from src.utils import load_fish_dataset

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
species = samples_df['specie'].values


# Create scaler: scaler
scaler = ____

# Create KMeans instance: kmeans
kmeans = ____

# Create pipeline: pipeline
pipeline = ____

# Fit the pipeline to samples
____

# Calculate the cluster labels: labels
labels = ____

# Create a DataFrame with labels and species as columns: df
df = ____

# Create crosstab: ct
ct = ____

# Display ct
print(ct)

