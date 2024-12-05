from sklearn.metrics import mean_squared_error
import numpy as np
from src.utils import load_new_dataframe_kc_house
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
house_df = load_new_dataframe_kc_house()

# Create X and y arrays
X = house_df["sqft_living"].values.reshape(-1,1)
y = house_df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X,y)

# Make predictions
y_pred = reg.predict(X)

plt.scatter(X,y)
plt.plot(X, y_pred, color='red')
plt.xlabel("Área interna da casa em pés quadrados")
plt.ylabel("preço")
plt.show()

#R-squared
r_squared = reg.score(X_test, y_test)

#RSS
rss = np.square(y-y_pred)

#MSE
mse = mean_squared_error(y, y_pred)

#RMSE
rmse = np.sqrt(mse)

# Print the metrics
print("RSS: {}".format(rss))
print("MSE: {}".format(mse))
print("RMSE: {}".format(rmse))
print("R^2: {}".format(r_squared))