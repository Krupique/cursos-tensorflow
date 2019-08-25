#Carregando base de dados
import pandas as pd
base = pd.read_csv('dados/house_prices.csv')

#Verificando somente os atributos que serão utilizados para a previsão
base.columns
colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']

#Carregando novamente a base de dados somente com os atributos úteis para a previsão
base = pd.read_csv('dados/house_prices.csv', usecols = colunas_usadas)

#Toda a parte de normalização
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])
base.head()

#Normalização dos preços
scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])

#Pegando toda a tabela sem o atributo preço
x = base.drop('price', axis = 1)
y = base.price #Pegando somente o atributo preço

previsores_colunas = colunas_usadas[1:17] #Colunas previsoras

#Adicionando as colunas ao tensorflow
import tensorflow as tf
colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]
#Separando dados em treino e teste
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

#Parte de treinamento.
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treino, y = y_treino, batch_size = 32, num_epochs = None, shuffle = True)
regressor = tf.estimator.DNNRegressor(hidden_units = [8, 8, 8], feature_columns = colunas)
regressor.train(input_fn = funcao_treinamento, steps = 20000)

#Parte de previsão
funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, shuffle = False)
previsoes = regressor.predict(input_fn = funcao_previsao)
list(previsoes)

#Adicionando somente a lista os valores previsões
valores_previsao = []
for p in regressor.predict(input_fn=funcao_previsao):
    valores_previsao.append(p['predictions'][0])

#"Desnormalizar"
import numpy as np
valores_previsao = np.asarray(valores_previsao).reshape(-1, 1)
valores_previsao = scaler_y.inverse_transform(valores_previsao)

y_teste2 = y_teste.values.reshape(-1, 1)
y_teste2 = scaler_y.inverse_transform(y_teste2)

#Calculando o Mean Absolute Error da previsão
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, valores_previsao)
mae