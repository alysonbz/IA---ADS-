# Import TSNE

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_grains_dataset

samples_df = load_grains_dataset()
samples = samples_df.drop(['variety','variety_number'],axis=1)
variety_numbers = samples_df['variety_number'].values


# Create a TSNE instance: model
model = __

# Apply fit_transform to samples: tsne_features
tsne_features =__

# Select the 0th feature: xs
xs = __

# Select the 1st feature: ys
ys =__

# Scatter plot, coloring by variety_numbers
___

plt.show()
