from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import Smart_Watch

Relogio_Inteligente = Smart_Watch()

# Tratar os valores ausentes (NaN)
Relogio_Inteligente.dropna(inplace=True)

# Converter variáveis categóricas em numéricas
le = LabelEncoder()
for coluna in Relogio_Inteligente.select_dtypes(include='object').columns:
    Relogio_Inteligente[coluna] = le.fit_transform(Relogio_Inteligente[coluna])

#separar obj target
X = Relogio_Inteligente.drop(columns=["Battery Life (days)"]).values
y = Relogio_Inteligente["Battery Life (days)"].values

correlacao = Relogio_Inteligente.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de Correlação")
plt.show()

# Ver a correlação direta com o alvo (Battery Life)
correlacao_target = correlacao["Battery Life (days)"].sort_values(ascending=False)
print("Correlação com Battery Life (days):\n", correlacao_target)

# Salvar o dataset modificado

Relogio_Inteligente.to_csv('Smart_Watch_Atualizado.csv', index=False)

print("Dataset modificado salvo como 'Smart_Watch_Atualizado.csv'")
