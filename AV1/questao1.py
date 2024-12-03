import pandas as pd

def load_water_quality():
    return pd.read_csv("../AV1/dataset/water_quality.csv")

waterQuality = load_water_quality()

print(waterQuality.shape)
print(waterQuality.isna().sum())
print(waterQuality)
