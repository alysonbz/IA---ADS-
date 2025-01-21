# Perform the necessary imports
import pandas as pd

from scipy.cluster.hierarchy import fcluster,linkage
from src.utils import load_movements_price_dataset
from sklearn.preprocessing import normalize

movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'],axis=1)
companies = movements_df['company'].values

normalized_movements = ____

# Calculate the linkage: mergings
mergings =___

# Use fcluster to extract labels: labels
labels = ___

# Create a DataFrame with labels and varieties as columns: df
df = __

# Create crosstab: ct
ct = __

# Display ct
print(ct)
