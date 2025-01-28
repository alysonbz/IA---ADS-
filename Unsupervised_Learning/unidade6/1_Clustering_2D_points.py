from src.utils import load_points
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

points = load_points()

# Create a KMeans instance with 3 clusters: model
model = __(__)

test_points = points[:50,:]
train_points = points[50:,:]

# Fit model to train_points
___

# Determine the cluster labels of new_points: labels
labels = __

# Print cluster labels of new_points
print(labels)


# Assign the columns of test_points: xs and ys
xs = test_points[:,0]
ys = test_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
___

# Assign the cluster centers: centroids
centroids =

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,s=50, marker = 'D')
plt.show()
