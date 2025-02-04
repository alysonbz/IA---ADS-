import pandas as pd

# Perform the necessary imports
from pandas import crosstab
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import load_fish_dataset

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
species = samples_df['specie'].values

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'specie': species})

# Create crosstab: ct
ct = crosstab(df['labels'], df['specie'])

# Display ct
print(ct)

