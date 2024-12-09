import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv('datasets/bodyfat.csv')

print("Informações do Dataset:")
print(dataframe.info())

correlation_matrix = dataframe.corr()

correlation_with_target = correlation_matrix["BodyFat"].sort_values(ascending=False)

print("\nAtributos mais correlacionados com BodyFat:")
top_features = correlation_with_target[1:6]
print(top_features)

plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features.index, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=dataframe[feature], y=dataframe["BodyFat"], alpha=0.7, color='b')
    plt.title(f"BodyFat vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("BodyFat")
plt.tight_layout()
plt.show()
