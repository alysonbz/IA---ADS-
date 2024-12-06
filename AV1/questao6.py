from sklearn.model_selection import train_test_split
from AV1.src.utils import load_car_price_prediction
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

carPrice = load_car_price_prediction()

X = carPrice["Gear box type"].values.reshape(-1, 1)
y = carPrice["Price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X, y)

y_pred = reg.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, color="red")
plt.xlabel("Gear box type")
plt.ylabel("Price")
plt.show()

r_squared = reg.score(X_test, y_test)
rss = np.square(y-y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"RSS: {rss}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r_squared}")
