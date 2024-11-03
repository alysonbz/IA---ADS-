# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 3

### Questão 1

```k-Nearest-Neighbors-Fit.py```

#### Classificação por KNN

Nesta questão você vai realizar uma predição com KNN.
#### Instruções:

1)  Importe `` KNeighborsClassifier `` .
   
2)  Obtenha as labels ``y`` com os valores da coluna ``churn``  do dataset churn_df. 

3)  Obtenha os atributos ``X`` com os dados ``account_length`` e  ``number_customer_service_calls`` do dataset churn_df.

4)  Inicialize o classificador KNN.  Atribua 6 vizinhos.
5) Execute o comando ``fit`` com os dados de ``X`` e ``y`` como argumentos.
6) Execute uma predição com a função ``predict`` do ``knn``. Atribua como argumento ``X_test``
7) Print as predições realizadas``y_pred``
8) 
### Questão 2

```2_evaluate_knn.py```
#### Avaliando KNN

Nesta questão você deve usar e avaliar o acerto do KNN

#### Instruções 

1) Importe o módulo ``train_test_split`` .
2) Divida o dataset com a função ``train_test_split``. Use ``X`` e ``y``, considere amostragem estratificada de acordo com ``y`` e conjunto de teste de 20% do tamanho do dataset. Atribua random state = 42.
3) Realize o comando fit, usando ``x_train`` e ``y_train``.
4) Print a acurácia com a função ``score``, usando ``X_teste`` e ``y_teste``.


### Questão 3

```3_underfiting_overfiting.py```

#### Dateaset com média zero e variância unitária

Nesta questão você vai realizar uma análise para verificar qual melhor k para uma classificação por knn.

#### Instruções

1) defina a quantidade de vizinhos para analisar. Considere uma variação de 1 a 12 vizinhos.
2) Inicialize o KNN com quantidade de vizinhos iterativa, variando de acordo com ``neighbor``.
3) Execute o comando ``fit`` com ``X_train`` e ``y_train``.
4) Utilizando ``neighbor`` como index alimente o dicionário de acurácias  com a função``score`` usando os dados de teste .
5) Use o título ``KNN: Varying Number of Neighbors``.
6) Execute o plot com ``neighbors`` e ``train_accuracies.values()``. Considere ``label = Training Accuracy``.
7)  Execute o plot com ``neighbors`` e ``test_accuracies.values()``. Considere ``label = Test Accuracy``.
8) Mostre o gráfico usando a função ``show()`` da ``pyplot``.