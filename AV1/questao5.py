from src.utils import load_new_dataframe_kc_house
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

houses = load_new_dataframe_kc_house()

X = houses.drop(["price"], axis=1)
y = houses["price"].values

#Grafico de correlação
correlation = houses.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Mapa de Correlação")
plt.show()

#Graficos padrão
houses = houses.drop(['sqft_lot', 'sqft_lot15', 'yr_built', 'condition'], axis=1)

sns.scatterplot(x='sqft_living', y='price', data=houses)
plt.show()

#LASSO: Esta demorando muito para criar o grafico. provavelmente pq tem muitas colunas.
sales_columns = X.columns

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.1)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()