from src.utils import load_new_dataframe_kc_house
import seaborn as sns
import matplotlib.pyplot as plt

houses = load_new_dataframe_kc_house()

print(houses['price'].values)

correlation = houses.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Mapa de Correlação")
plt.show()

houses = houses.drop(['sqft_lot', 'sqft_lot15', 'yr_built', 'condition'], axis=1)

sns.scatterplot(x='sqft_living', y='price', data=houses)
plt.show()

