# Import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_movements_price_dataset
from sklearn.preprocessing import normalize

# Create a TSNE instance: model
model =  __

movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'],axis=1)
companies = movements_df['company'].values
normalized_movements = normalize(movements)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = ____

# Select the 0th feature: xs
xs = ____

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
____

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
