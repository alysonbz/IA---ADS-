import pandas as pd
from pandas import isnull


def load_gender_classification():
    return pd.read_csv('dataset/gender_classification_v7.csv')

def load_kc_house_data():
    return pd.read_csv('dataset/kc_house_data.csv')

def load_new_dataframe_kc_house():
    return load_kc_house_data().drop(['long', 'lat', 'zipcode', 'date', 'id'], axis=1)

def load_new_dataframe_gender_classification():
    data = load_gender_classification()
    data["gender"] = data["gender"].map({'Male':1 ,'Female':0})
    return data
#Vou colocar essas Funçoes no arquivo utils para ajudar nas proximas questões.

dataclassification = load_gender_classification()
dataRegressao = load_kc_house_data()
newDataRegressao = load_new_dataframe_kc_house()
newDataClassification = load_new_dataframe_gender_classification()

#Printes verificando se tem NA ou null, Não tem.
#print(dataRegressao.isna().sum(),'\n'), print(dataclassification.isna().sum(),'\n'), print(dataRegressao.isnull().sum(),'\n'), print(dataclassification.isnull().sum(),'\n')

#prints para verificar os tipos dos arquivos, Percebi que no gender_classification, Gender era objeto então transformei
    #em numerico em que 1 é masculino e 0 é feminino.
#print(dataclassification.dtypes), print(dataRegressao.dtypes)

#print para verificar se ta tudo certo
print(newDataClassification.dtypes), print(newDataRegressao.value_counts())