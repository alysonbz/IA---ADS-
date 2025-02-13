import pandas as pd
from pandas import crosstab

from src.utils import load_gender_classification
from sklearn.cluster import KMeans

gender_df = load_gender_classification()

samples = gender_df[['nose_wide','distance_nose_to_lip_long']]

gender = gender_df['gender'].values

model = KMeans(n_clusters=2)

labels = model.fit_predict(samples)

df = pd.DataFrame({'labels': labels, 'varieties': gender})

ct = crosstab(labels, gender)

print(ct)