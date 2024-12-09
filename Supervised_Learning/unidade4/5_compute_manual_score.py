import numpy as np
from sklearn.metrics import mean_squared_error

from src.utils import processing_all_features_sales_clean

def compute_RSS(predictions,y):
    RSS = np.sum(np.square(y - predictions))
    return RSS
def compute_MSE(predictions,y):
    MSE= np.mean(np.square(y - predictions))
    return MSE
def compute_RMSE(predictions,y):
    RMSE = np.sqrt(compute_MSE(predictions - y))
    return RMSE
def compute_R_squared(predictions,y):
    total_variance = np.sum(np.square(y - mean(y)))
    residuals_variancia = 1 - (residuals_variancia / total_variance)
    return residuals_variancia


X,y,predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))