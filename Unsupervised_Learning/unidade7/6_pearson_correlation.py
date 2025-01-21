# Perform the necessary imports
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from src.utils import load_grains_dataset


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = __

# Assign the 1st column of grains: length
length = __

# Scatter plot width vs length
plt.scatter(____, ____)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = ____

# Display the correlation
print(correlation)
