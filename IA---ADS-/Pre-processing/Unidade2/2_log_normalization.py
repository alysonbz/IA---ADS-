import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

#print as caractéristicas estatísticas do dataset wine
print(wine.__)

## Aplique a função de nomarlização logarítmica na coluna Proline
wine[__] = np.__(___)
#
# Print a variância da coluna proline
print(___)

# print a variância da coluna proline normalizada
print(np.var(wine['Proline_log']))