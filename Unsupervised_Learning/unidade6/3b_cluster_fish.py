import pandas as pd
from src.utils import load_fish_dataset
from sklearn.cluster import KMeans

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
specie = samples_df['specie'].values

# Create KMeans instance: kmeans with 4 custers


# Use fit_predict to fit model and obtain cluster labels: labels


# Create a DataFrame with labels and varieties as columns: df


# Create crosstab: ct

# Display ct