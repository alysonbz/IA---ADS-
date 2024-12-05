import numpy as np
from src.utils import processing_all_features_sales_clean


def compute_RSS(predictions, y):
    # Diferenças quadráticas somadas
    RSS = np.sum(np.square(y - predictions))
    return RSS
def compute_MSE(predictions,y):
    RSS = compute_RSS(predictions, y)
    MSE = np.divide(RSS, len(predictions))
    return MSE
def compute_RMSE(predictions,y):
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE
def compute_R_squared(predictions, y):
    ss_total = np.sum(np.square(y - np.mean(y)))  # Soma dos quadrados totais
    ss_residual = compute_RSS(predictions, y)    # Soma dos quadrados residuais
    r_squared = 1 - (ss_residual / ss_total)     # Fórmula do R^2
    return r_squared


X,y,predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))