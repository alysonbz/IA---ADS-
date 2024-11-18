from src.utils import processing_sales_clean
# Importar o pyplot da biblioteca matplotlib
import matplotlib.pyplot as plt

# Carregar dados e previsões
X, y, predictions = processing_sales_clean()

# Criar um gráfico scatter utilizando X e y, com a cor azul
plt.scatter(X, y, color="blue")

# Criar um gráfico de linha utilizando X e predictions, com a cor vermelha
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")


# Mostrar o gráfico
plt.show()