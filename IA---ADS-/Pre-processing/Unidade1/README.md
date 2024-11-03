# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 1

### Questão 1

```1_pre_processing.py```

#### Pré-processamento

Neste primeiro exercício você deve realizar manipulação em um dataset com a biblioteca pandas.

#### Instruções:

1)  Utilizando a função `` shape `` do objeto `` volunteer `` print o tamanho do dataset.
   
2)  Utilizando a função `` info`` do objeto `` volunteer `` print as caracteristicas do dataset

3)  Print a quantidade de elementos que estão faltando na coluna `` locality``.
    
4)  Exclua as colunas ``Latitude`` e ``Longitude`` de volunteer e coloque em um dataframe ``volunteer_cols``

5)  Exclua as linhas com valores null da coluna ``category_desc`` de ``volunteer_cols`` e coloque em um dataframe ``volunteer_subset``

6) print a dimensão de Print o shape de ``volunteer_subset``

### Questão 2

```2_data_types.py```

#### Instruções 

1) Print os 5 primeiros elementos da coluna ``hits`` do dataset ``volunteer``.
2) Print as caracteristicas da coluna``hits``.
3) Converta a coluna ``hits`` de ``volunteer`` para o tipo ``int32``.
4) Print as caracteristicas da coluna``hits`` novamente.


### Questão 3

```3_training_and_test_sets.py```

#### Treinando um dataset

Neste exercicio você irá aprender a separar o dataset em um conjunto para treino e teste.

#### Instruções

1) importe a função ``train_test_split``
2) Exclua as colunas ``Latitude`` e ``Longitude`` de volunteer com a função ``drop``.
3) Exclua as linhas com valores ``null`` da coluna ``category_desc`` de ``volunteer_new``
4) Mostre o balanceamento das classes em ``category_desc`` utilizando a função ``value_counts``
5) Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
6) Crie um dataframe de labels com a coluna ``category_desc``
7) Utiliza a a amostragem stratificada para separar o dataset em treino e teste
8) Mostre o balanceamento das classes em ``category_desc`` novamente
