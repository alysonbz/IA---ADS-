# Perform the necessary imports
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from src.utils import load_movements_price_dataset
from sklearn.preprocessing import normalize

movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'], axis=1)
companies = movements_df['company'].values

normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and companies as columns: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['companies'])

# Display ct
print(ct)
