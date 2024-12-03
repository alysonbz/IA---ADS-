# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 5

### Questão 1

[1_confusion_matrix.py](1_confusion_matrix.py)

#### Matrix de confusão 

Nesta questão você vai calcular a matriz de confusão e um relatório de métricas.

#### Instruções:

1)  Importe o módulo para calcular a matriz de confusão e o classification report.
   
2)  Execute a função fit do knn da forma apropriada. 

3)  Realize uma predição com a função predict usando o conjunto de teste.

4)  Print a matriz de confusão e o classification report.


### Questão 2

[2_compute_metric_manual.py](2_compute_metric_manual.py)

#### Clacular manualmente as métricas

Nesta questão você vai calcular a matriz de confusão e um relatório de métricas manual.

#### Instruções:

1)  Calcule a acurácia geral
   
2)  Informe o recall de cada classe. 

3)  Informe precision para cada classe.

4)  Exiba a matriz de confusão

### Questão 3

[3_logistic_regression.py](3_logistic_regression.py)

#### Classificação via regressão logística

Nesta questão você vai classificar dados via regressão logística.

#### Instruções:

1)  importe o moudulo de regressão logística
   
2)  Instancie o modelo de regressão logística. 

3)  treine o modelo de regressão logistica.

4)  calcule as probabilidades com a função ``predict_proba``, armazenando somente a probabilidade de haver diabetes.

### Questão 4

[4_roc_curve.py](4_roc_curve.py)

#### Plotando a curva ROC

Nesta questão você vai plotar a curva ROC.

#### Instruções:

1)  importe o moudulo ``roc_curve``
   
2)  gere a curva roc com a função ``roc_curve``. 

3)  plote a curva roc.


### Questão 5

[5_auc.py](5_auc.py)

#### Calculando AUC

Nesta questão você vai calcular o AUC

#### Instruções:

1)  importe o moudulo para calcula AUC
   
2)  print o valor do AUC

3)  print a matriz de confusão.

4)  print ``classifiction_report``.

6) Faça um relatório comparando o desempenho da classificação do KNN e Regressão logistica para este dataset.
modelo de relatório:
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=112368782396865447257&rtpof=true&sd=true

### Questão 6

[6_hyperparam_tuning_grid_search.py](6_hyperparam_tuning_grid_search.py)

#### Análise de hiperparamentros com grid search

Nesta questão você vai buscar a melhor paramentrização para um modelo de regressão.

#### Instruções:

1)  importe o moudulo de regressão ``Lasso``
   
2)  importe o modulo ``kfold``. 

3)  importe o modulo gridsearch.

4) Inicialize o regressor Lasso

5) Inicialize o kfold consideranto 5 pastas, embaralhamento e ``random_state = 42``.

5) Crie a variável ``param_grid`` , considerando conjunto de 20 variáveis ``"alpha"`` variando de 0.00001 até 1.

6) Inicialize o gridsearch considerando os argumentos:  lasso, param_grid e cv=kf.

7) Treine o regressor com o conjunto de treino.




### Questão 7

[7_hyperparam_tunig_random_search.py](7_hyperparam_tunig_random_search.py)

#### Análise de hiperparamentros com randomized search

Nesta questão você vai buscar a melhor parametrização para um modelo de classificação.

#### Instruções:

1)  importe o moudulo ``LogisticRegression``
   
2)  importe o modulo ``kfold``. 

3)  importe o modulo RadomizedSearch.

4) Inicialize LogistcRegression

5) Inicialize o kfold consideranto 5 pastas, embaralhamento e ``random_state = 42``.

5) Crie a variável ``param`` , considerando: "penalty": ["l1", "l2"], "tol": np.linspace(0.0001, 1.0, 50),"C": np.linspace(0.1, 1, 50) e "class_weight": ["balanced", {0:0.8, 1:0.2}]
         
6) Inicialize o RandomizedSearchCV considerando os argumentos:  logreg, params e cv=kf

7) Treine o modelo com o conjunto de treino.

8) Mostre a melhor parametrização

9) Mostre a melhor acurácia


