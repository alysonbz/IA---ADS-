import pandas as pd

def load_water_quality_old():
    return pd.read_csv("dataset/water_quality.csv")

def load_car_price_prediction():
    return pd.read_csv("dataset/car_price_prediction.csv")

def load_water_quality():
    return load_water_quality_old().dropna(subset=['ammonia', 'is_safe'])

