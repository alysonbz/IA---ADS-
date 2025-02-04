# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 6

### Questão 1

[1_Clustering_2D_points.py](1_Clustering_2D_points.py)

#### Clusterização de pontos 2D

Nesta questão você vai agrupar pontos com o K-means

#### Instruções:

1)  Inicialize o k-means com 3 clusters.
   
2)  Execute a função fit do para ``train_points``. 

3)  Realize uma predição com a função predict em ``test_points`` e armazene em labels

4)  Faça um scatter plot para relacionar xs e ys. Use ``labels`` para definir as cores e faça alpha = 0.5.

5) Use a função ``cluster_centers_`` do objeto model para obter os 3 pontos centroides e armazene a resposta em centroids.


### Questão 2

[2_inertia.py](2_inertia.py)

#### Método do Cotovelo

Nesta questão você vai encontrar o k adequado para o kmeans.

#### Instruções:

1)  Inicialize o k-means com clusters iterativos k, criando o objeto model.
   
2)  Execute a função fit de model para ``samples``. 

3)  Calcule a inertia a adicione o valor na lista``inertias`` com a função ``append``.


### Questão 3

[3_crosstab.py](3_crosstab.py)

#### Observação dos clusters em tabela

Nesta questão você vai analisar se a clusterização ocorreu adequadamente analisando a tabela

#### Instruções:

1)  Inicialize o k-means com 3 clusters, criando o objeto model.
   
2)  Execute a função ``fit_predict`` de model para ``samples`` e armazene o resultado em labels. 

3)  Crie um ``crosstab ``, relacionando ``labels`` e ``varieties``.

4) No arquivo [4_cluster_fish.py](3b_cluster_fish.py) repita essa mesma questão e compare as clusterizações. Considere 4 clusters

### Questão 3

[3_crosstab.py](3_crosstab.py)

#### Observação dos clusters em tabela

Nesta questão você vai analisar se a clusterização ocorreu adequadamente analisando a tabela

#### Instruções:

1)  Inicialize o k-means com 3 clusters, criando o objeto model.
   
2)  Execute a função ``fit_predict`` de model para ``samples`` e armazene o resultado em labels. 

3)  Crie um ``crosstab ``, relacionando ``labels`` e ``varieties``.

4) No arquivo [3b_cluster_fish.py](3b_cluster_fish.py) repita essa mesma questão e compare as clusterizações. Considere 4 clusters



### Questão 4

[4_pipeline.py](4_pipeline.py)

#### Criação de pipilines 

Nesta questão você vai realizar um pipiline de pre-processamento e clusterização.

#### Instruções:

1) importe os módulos ``make_pipeline``, ``StandardScaler`` e  ``KMeans``.
   
2)  Instancie o StandardScaler no objeto scale. 

3) instancie o kmeans com 4 clusters

4) Instancie o pipeline com o modo make_pipeline adicionando os argumentos scale e kmeans. 

5) Utilize a função fit de pipeline no conjunto de dados samples

6) Utilize a função predict de pipeline no conjunto de dados samples

7) Crie um dataframe associando labels e species

8) crie um crosstab associando labels e species


### Questão 5

[5_normalizer.py](5_normalizer.py)

#### Normalização das linhas

Nesta questão você vai realizar um pipiline de pre-processamento e clusterização nomrlaizando as linhas.

#### Instruções:

1) importe os módulos ``make_pipeline``, ``Normalizer`` e  ``KMeans``.
   
2)  Instancie o Normalizer no objeto normalizer. 

3)  Instancie o kmeans com 10 clusters

4) Instancie o pipeline com o modulo make_pipeline adicioanando os argumentos normalize e kmeans. 

5) Utilize a função fit de pipeline no conjunto de dados movements

6) Utilize a função predict de pipeline no conjunto de dados movements

7) Crie um dataframe associando labels e companies

8) Utilizando a função ``.sort_values`` de df mostre a coluna ``labels`` do dataframe criando no item 7 de forma ordenada.


