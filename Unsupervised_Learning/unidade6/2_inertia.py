import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

samples_df = load_grains_dataset()
samples = samples_df.drop(['variety','variety_number'],axis=1)

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    ____

    # Fit model to samples
    ____

    # Append the inertia to the list of inertias
    ____

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
