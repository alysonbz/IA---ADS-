# Import pandas
import pandas as pd
# Import Normalizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from src.utils import load_movements_price_dataset

movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'],axis=1)
companies = movements_df['company'].values

# Create a normalizer: normalizer
normalizer = ____

# Create a KMeans model with 10 clusters: kmeans
kmeans = ____

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = ____

# Fit pipeline to the daily price movements
____

# Predict the cluster labels: labels
labels = __

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print()
