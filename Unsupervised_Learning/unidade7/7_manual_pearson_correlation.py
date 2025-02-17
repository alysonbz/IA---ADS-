# Perform the necessary imports
import numpy as np

from src.utils import load_grains_dataset


def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


grains_df = load_grains_dataset()

# Assign the 0th column of grains: width
width = grains_df.iloc[:, 0]

# Assign the 1st column of grains: length
length = grains_df.iloc[:, 1]

# Calculate the Pearson correlation
correlation = pearson_correlation(width, length)

# Display the correlation
print(correlation)
