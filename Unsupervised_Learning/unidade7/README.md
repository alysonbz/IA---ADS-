# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 7

### Questão 1

[1_dendogram.py](1_dendogram.py)

#### Gerando um Dendograma

Nesta questão você vai agrupar pontos com dendograma

#### Instruções:

1)  Importe o módulo dendogram e linkage.
   
2)  Execute a função linkage e amramzene o resultado em mergings. 

3) plot o dendogram e enontre a melhor parametrização para leaf_rotation e leaf_font_size.


### Questão 2

[2_dedogram_movements.py](2_dedogram_movements.py)

#### Dendograma de dados normalizados

Nesta questão você visualizar dendograma de dados normalizados

#### Instruções:

1)  Normalize os dados movements com a função Normalizer da scikit-learn.
   
2)  Execute a função linkage nos dados normalizdos e armamzene o resultado em mergings. 

3) plot o dendogram e enontre a melhor parametrização para leaf_rotation e leaf_font_size.

### Questão 3

[3_fcluster.py](3_fcluster.py)

#### Observação dos clusters em tabela

Nesta questão você vai analisar se a clusterização ocorreu adequadamente analisando a tabela

#### Instruções:

#### Instruções:

1)  Normalize os dados movements com a função Normalizer da scikit-learn.
   
2)  Execute a função linkage nos dados normalizdos e armamzene o resultado em mergings. 

3)  Use a função fcluster para que seja possível gerar 6 cluters neste dataset. Observe o dendograma da questão anterior.

4) Crie um dataframe relacionando labels e companies

5) crie um crosstab relacioando labels e companies

### Questão 4

[4_tsne.py](4_tsne.py)

#### T-SNE

Nesta questão você vai reduzir o dataset para duas dimensões e plotar os atributos

#### Instruções:

#### Instruções:

1)  Instancie o T-SNE com learning_rate igual a 200.
   
2)  Execute a função fit_transform no conjunto de dados samples 

3)  Selecione a primeira coluna de tsne_features

4)  Selecione a segunda coluna de tsne_features

5)  Crie um scatter plot indicando a cor pelas labels variety_numbers


### Questão 5

[5_tsne_2.py](5_tsne_2.py)

#### T-SNE 2

Nesta questão você vai reduzir o dataset para duas dimensões e plotar os atributos

#### Instruções:

#### Instruções:

1)  Instancie o T-SNE com learning_rate igual a 50.
   
2)  Execute a função fit_transform no conjunto de dados normalized_movements 

3)  Selecione a primeira coluna de tsne_features

4)  Selecione a segunda coluna de tsne_features

5)  Crie um scatter plot indicando alpha igual a 0.5


### Questão 6

[6_pearson_correlation.py](6_pearson_correlation.py)

#### Correlação de Pearson

Nesta questão você vai aplicar o calculo da correlação de pearson


#### Instruções:

1)  Armazene em width a primeira coluna do dataset grains.
   
2)  Armazene em lenght a segunda coluna do dataset grains 

3)  Crie um scatter plot de width em função de lenght

4)  calcule a correlação de pearson utilizando a função importada pearsonr. Esta função recebe dois argumentos para calcular a correlçao entre estes.



### Questão 7

[7_manual_pearson_correlation.py](7_manual_pearson_correlation.py)

#### Correlação de Pearson

Nesta questão você vai aplicar o calculo da correlação de pearson de forma manual


#### Instruções:

1)  Armazene em width a primeira coluna do dataset grains.
   
2)  Armazene em lenght a segunda coluna do dataset grains 

3)  Crie um scatter plot de width em função de lenght

4)  calcule a correlação de pearson utilizando a função que você deve implementar. Esta função recebe dois argumentos para calcular a correlçao entre estes e retorna a correlação.


### Questão 8

[8_PCA_decorrelation.py](8_PCA_decorrelation.py)

#### Correlação de Pearson

Nesta questão você vai aplicar o calculo da descorrelação com o método PCA.


#### Instruções:

1)  Instancie o modelo do PCA.
   
2)  Execute a função fit_transform para gerar os atributos descorrelacionados 

3) Armazene em xs a primeira coluna de pca_features e em ys a segunda coluna.
4) Crie um scatter plot de xs em função de ys.
5)  Calcule a correlação de pearson.
6) Mostre o valor da correlação.

### Questão 9

[9_pca_variance_analysis.py](9_pca_variance_analysis.py)

#### Dimensão Intríseca

Nesta questão você vai utilizar o PCA para determinar a dimensão intríseca.


#### Instruções:

1)  Instancie o standardscaler para normalizar os dados   
2)  Instancie o PCA 
3)  Crie um pipiline para normalizar e aplicar o pca.
4)  Execute a função fit do pipiline no conjunto sampples.
5)  Plote as variâncias.

### Questão 10

[10_pca_dimention_reduction.py](10_pca_dimention_reduction.py)

#### Redução da dimensão

Nesta questão você vai utilizar o PCA para reduzir a dimensão


#### Instruções:

1)  Instancie o PCA com a quantidade de componentes que possuem variancia superior a 1.5 da questão 9
2)  Treine o PCA com os dados normalizados
3)  Print o shape dos novos atributos do PCA.
4)  Visualize os novos atributos em duas dimensões com scatter plot. Utilize as labels das classes para definir as cores.

### Questão 11

[10_pca_dimention_reduction.py](10_pca_dimention_reduction.py)

#### Aplicação da aprendizagem não supervisionada

Nesta questão você vai utilizar o PCA para reduzir a dimensão


#### Instruções:

1) Faça uma redução da dimensão com PCA do dataset e escolha um método para classificar os atributos 
gerados pelo PCA obstidos na questão 10. Calcule a acurácia, gere o classification report e a matriz de confusão.