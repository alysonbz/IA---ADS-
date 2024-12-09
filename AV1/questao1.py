from AV1.src.utils import load_water_quality_old, load_water_quality

waterQuality = load_water_quality_old()
newWaterQuality = load_water_quality()

print(waterQuality.shape)
print(waterQuality.isna().sum())
print(waterQuality.dtypes)
print(waterQuality)
print(newWaterQuality.isna().sum())
print(newWaterQuality.dtypes)
print(newWaterQuality)
