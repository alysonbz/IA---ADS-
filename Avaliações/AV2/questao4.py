from Avaliações.AV1.src.utils import load_water_quality
from sklearn.cluster import KMeans
import pandas as pd

water_df = load_water_quality()

samples = water_df[['aluminium', 'chloramine']]
water = water_df['is_safe'].values

model = KMeans(n_clusters=5)

labels = model.fit_predict(samples)

df = pd.DataFrame({
    'labels': labels,
    'varieties': water
})

print(pd.crosstab(labels, water))
