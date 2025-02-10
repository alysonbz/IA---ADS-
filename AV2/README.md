# AVALIAÇÃO 2 - Prazo para envio do relatório e código : 17/02/2025
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

ANTONIO LUCAS MELO DE SOUSA: 

https://www.kaggle.com/datasets/erdemtaha/cancer-data

CAUA DE SOUSA BRANDAO: 

 https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset



ESDRAS SOUZA DOS SANTOS : 

 https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset



GUILHERME PINHEIRO SERAFIM: 

 https://www.kaggle.com/datasets/whenamancodes/predict-diabities



JOAO LUIS FEITOSA LEITE: 

 https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings



JOAO LUIS FEITOSA LEITE: 

https://www.kaggle.com/datasets/prathamtripathi/drug-classification



JOSE DAVI ARAUJO GOMES: 

https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset



JOSE ERIC MESQUITA COELHO: 

 https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17


LUAN SALES BARROSO: 

 https://www.kaggle.com/datasets/mssmartypants/water-quality


MATHEUS FEITOSA DE OLIVEIRA RABELO:

https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification



ROBERT MICHAEL DE AVILA BARREIRA:

 https://www.kaggle.com/datasets/yasserh/wine-quality-dataset


SUZIANE BRANDAO ANDRADE:

 https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset



VITOR BARBOSA DA SILVA:

https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023






### Questão 1

```questao1.py```

Na AV1 você realizou uma análise do KNN para descobrir o melhor valor de k e também a forma mais adequada de determinar
a metrica de distância. Utilando o GridSearchCV que está disponível na scikit-learn, realize o mesmo processo para descobrir
qual a melhor parametrização no processo de classificação do seu dataset. Gere os deados de acertos completos e faça uma análise
do resultado. Verifique se o resultado obtido na AV1 é o mesmo que você obteve nesta AV2.


### Questão 2

```questao2.py```
Sem considerar a coluna alvo do seu dataset, faça uma análise para descobrir a quantidade ideal de clusters que deve ser 
feita. Considerem usar o o metodo do cotovelo e da silhueta. 

### Questão 3

```questao3.py```

Escolha os dois atributos mais relevantes utilizando o metodo de Lasso
e recalcule a quantidade de clusters com método do cotovelo e silhueta. Mudou a quantidade de cluster em relação aquestão anterior ? Se k diferentes para os dois metodos de deteminação da quantidade de clusters,
Faça dois scatterplots e faça uma análise visual.


### Questão 4

```questao4.py```

Faça um crosstab para que possamos identificar como ficou a distribuição de cluster de acordo com classes presentes na coluna 
alvo do seu dataset. Utilize o K means e utilize o k obtido pelo indice de sulhueta.


### Observações para o Relatório

Discutir **organizadamente** na sessão de resultados os números obtidos de cada questão.
Ao concluir o relatório, realizar o push na sua branch nesta mesma pasta.