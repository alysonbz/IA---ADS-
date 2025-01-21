import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

samples_df = load_grains_dataset()
samples = samples_df.drop(['variety','variety_number'],axis=1)
varieties = samples_df['variety'].values

# Create a KMeans model with 3 clusters: model
model = ____

# Use fit_predict to fit model and obtain cluster labels: labels
labels = ____

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = ____

# Display ct
print(ct)
