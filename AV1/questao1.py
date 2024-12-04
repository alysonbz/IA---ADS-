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

'''
aluminium - perigoso se for maior que 2.8
ammonia - perigoso se for maior que 32.5
arsenic - perigoso se for maior que 0.01
barium - perigoso se for maior que 2
cadmium - perigoso se for maior que 0.005
chloramine - perigosa se for maior que 4
chromium - perigoso se for maior que 0.1
copper - perigoso se for maior que 1.3
flouride - perigoso se for maior que 1.5
bacteria - perigosas se forem maiores que 0
viruses - perigoso se forem maiores que 0
lead - perigoso se forem maiores que 0.015
nitrates - perigoso se forem maiores que 10
nitrites - perigoso se forem maiores que 1
mercury - perigoso se forem maiores que 0.002
perchlorate - perigoso se forem maiores que 56
radium - perigoso se forem maiores que 5
selenium - perigoso se forem maiores que 0.5
silver - perigoso se forem maiores que 0.1
uranium - perigoso se forem maiores que 0.3
is_safe - atributo de classe {0 - não seguro, 1 - seguro}
'''
