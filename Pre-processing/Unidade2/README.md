# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 2

### Questão 1

```1_standardization.py```

#### Classificar sem normalizar

Nesta questão você vai realizar uma classificação com KNN sem executar normalizaçao dos atributos.

#### Instruções:

1)  Utilizando a função `` train_test_split `` Divida o dataset em dados para teste e dados para treino. Faça uma amostragem estratificada com base na proporção de label de ``y``.
   
2)  Execute o treinamento do classifcador knn. Este já foi iniccalizado para você com a linha  `` knn = KNeighborsClassifier() ``. Agora você deve chamar a função ``fit`` e colocar como argumentos ``X_train`` e ``y_train``. 

3)  Execute a função ``score`` de ``knn`` para medir a acurácia do classidicador. 

### Questão 2

```2_log_normalization.py```
#### Normalização Logarítimica

Nesta questão você deve observar os efeitos da normalização logarítimica

#### Instruções 

1) Utilizando a função ``describe()`` verifique as características estatística do dataset ``wine``.
2) Na coluna``Proline``, aplique a normalizaçao logarítmica e atribua o resultado para uma nova coluna, ``Proline_log``.
3) Exiba a variância da coluna ``Proline``.
4) Print a variância da coluna ``Proline_log``.


### Questão 3

```3_scaling_data.py```

#### Dateaset com média zero e variância unitária

Nesta questão você poder normalizar o dataset para uma média zero e variância unitária.

#### Instruções

1) Importe ``StandardScaler`` da scikit-learn
2) Inicialize o normalizador ``scaler``.
3) Como vamos classificar a qualidade do vinho, remova do dataset a coluna exclua do dataset a coluna ``Quality`` e armazene o resultado em ``X``.
4) Aplique a função ``fit_transform`` de ``scaler`` para obter o dataset normalizado e armazene o resultado em ``X_norm``.
5) Em ``y`` aramazene somente os valores da coluna ``Quality`` que serão nossas labels.
6) Print a variância de ``X``
7) Prnt a variância de ``X_norm``
8) Divida o dataset em treino e teste com amostragem estratificada usando ``train_test_split`` novamente. Lembre de usar ``X_norm`` como primeiro argumento.
9) Inicialize o classificador KNN
10) Aplique a função ``fit`` do knn com ``X_train`` e ``y_train``.
11) Mostre o acerto do algorito com a função ``score``.