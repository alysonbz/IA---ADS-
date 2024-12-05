import pandas as pd

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
print("Verificando se tem NA no dataset de regressão: ",dataRegressao.isna().sum(),'\n')
print("Verificando se tem NA no dataset de classificação: ",dataclassification.isna().sum(),'\n')
print("Verificando se tem Null no dataset de regressão: ",dataRegressao.isnull().sum(),'\n')
print("Verificando se tem Null no dataset de classificação: ",dataclassification.isnull().sum(),'\n')

#prints para verificar os tipos dos arquivos, Percebi que no gender_classification, Gender era objeto então transformei
#em numerico em que 1 é masculino e 0 é feminino.
print("Tipose de Dados do dataset de classificação: ",dataclassification.dtypes), print("Tipo de Dados do dataset de Regressão: ", dataRegressao.dtypes)

#print para verificar se ta tudo certo
print("Dataset classificação depois das mudanças: ",newDataClassification.dtypes), print("Dataset Regressão depois das mudanças: ",newDataRegressao.value_counts())