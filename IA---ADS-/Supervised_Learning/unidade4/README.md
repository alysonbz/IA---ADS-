# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 4

### Questão 1

[1_create_features.py](1_create_features.py)

#### Introdução a regressão Linear

Nesta questão você vai gerar os atributos para treinar um modelo de regressão.

#### Instruções:

1)  Armazene na variável ``X`` os dados da coluna ``radio`` do dataframe ``sales_df`` .
   
2)  Obtenha as labels ``y`` com os valores da coluna ``sales``  do dataframe ``sales_df``. 

3)  Redimensione ``X`` para estrutura apropriada de regressão utilizando 1 atributo somente.

4)  Observe as dimensões de X e de y.

### Questão 2

[2_build_regression.py](2_build_regression.py)

#### Criando um regressor

Nesta questão você deve usar o modelo de regressão para realizar predição.

#### Instruções 

1) Importe o modelo ``LinearRegression`` da scikit learn.
2) Inicialize o modelo.
3) Execute o método ``fit`` de ``reg``, usando ``X`` e ``y``.
4) Execute o método ``predict`` de ``reg``, usando ``X``.
5) Print as 5 primeias predições contidas em ``predictions``


### Questão 3

[3_visualize_predictions_resp.py](3_visualize_predictions_resp.py)

#### Visualizando as predições via regressão linear

Nesta questão você vai poder visualizar as predições em forma de gráfico.

#### Instruções

1) Importe a ``pyplot``da ``matplotlib`` como ``plt``
2) Crie um gráfico do tipo scatter utilizando ``X`` e ``y``. Utilize a cor azul.
3) Crie um gráfico do tipo linha utilizando ``X`` e predictions.Utilize a cor vermelha.
4) Utilize o método ``show`` de plt para mostrar o gráfico. 


### Questão 4

[4_fit_prediction.py](4_fit_prediction.py)

#### Avaliação das predições de uma regressão linear

Nesta questão você vai calcular os erros da regressão

#### Instruções

1) Importe a raiz do erro quadrático médio  ``mean_squared_error``
2) Exclua as colunas  ``sales`` e ``influencer`` e armazene o resultado em X.
3) Armazene os valores da coluna ``sales`` em y.
4) Iniciaize o modelo de regressão.
5) Calcule o coeficiente de determinação.
6) Calcule o valor da raís do erro quadrático médio.

### Questão 5

[5_compute_manual_score.py](5_compute_manual_score.py)

#### Avaliação das predições de uma regressão linear

Nesta questão você vai poder visualizar as predições em forma de gráfico.

#### Instruções

1) Com auxílio somente da lib ``numpy`` calcule dentro da função ``compute_RSS`` a soma dos quadrados dos resíduos
2) Com auxílio somente da lib ``numpy`` calcule dentro da função ``compute_MSE`` erro quadrático médio
3) Com auxílio somente da lib ``numpy`` calcule dentro da função ``compute_RMSE`` a raíz do erro quadrático médio
4) Com auxílio somente da lib ``numpy`` calcule dentro da função ``compute_R_squared``  o coeficiente de determinação.

### Questão 6

[6_cross_validadation_r_2.py](6_cross_validadation_r_2.py)

#### Validação cruzada

Nesta questão você vai poder realizar uma validação cruzada

#### Instruções

1) Importe as funções ``cross_val_score`` e ``KFold`` da ``scikit-learn``.
2) Inicialize ``KFold`` com k=6 com embaralhamento setado True.
3) Execute a função ``cross_val_score`` paramentrizando com o objeto reg, X,y e o objeto kf. 
4) Print o cv_scores.
5) print a média de cv_scores
6) print o desvio padrão de cv_scores


### Questão 7

[7_ridge_regularization.py](7_ridge_regularization.py)

#### Regularização de Ridge

Nesta questão você vai poder usar regularização de ridge.

#### Instruções

1) Importe ``Ridge``  da  ``scikit-learn``.
2) Inicialize ``Ridge`` alpha iterativo variando conforme o loop descrito.
3) Execute a função ``fit`` de ``Ridge``, parametrizando com X_train e y_train. 
4) Calcule o coeficiente de determinação R-squared computando a função ``score`` do objeto ridge


### Questão 8

[8_lasso_regression.py](8_lasso_regression.py)

#### Lasso

Nesta questão você vai poder determinar os atributos mais importantes para uma possível regressão.

#### Instruções

1) Importe ``Lasso``  da  ``scikit-learn``.
2) Inicialize ``Lasso`` com  alpha=0.3.
3) Execute a função ``fit`` de ``lasso``, calculando os coeficientes. 


