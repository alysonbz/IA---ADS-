from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize um print do dataset volunteer corrigido sem nenhum NAN, removam as colunas NAN e depois as linhas e crie
##um dataframe novo

volunteer = volunteer.dropna [''] (axis =1)