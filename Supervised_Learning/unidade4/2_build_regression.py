from src.utils import load_sales_clean_dataset

#Carregar o dataset
sales_df = load_sales_clean_dataset()

#1
from sklearn.linear_model import LinearRegression

#Extrair os dados
y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

#2
reg = LinearRegression()

#3
reg.fit(X, y)


#4
predictions = reg.predict(X)

#5
print(predictions[:5])