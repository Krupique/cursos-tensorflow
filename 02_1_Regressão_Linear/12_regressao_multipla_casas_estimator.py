#Ciclo de execução com estimators:
"""
    1 - Dados
    2 - Criar funções input
    3 - Passar para o estimator

    4 - Criar função para fazer os treinamentos
    5 - Criar função para fazer as avaliações
    6 - Croar função para fazer as previsões
"""

#Quando trabalhamos com tensorflow podemos usar tanto pandas quanto numpy
import pandas as pd
base = pd.read_csv('dados/house_prices.csv')
base.head()

base.columns
colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']
base = pd.read_csv('dados/house_prices.csv', usecols = colunas_usadas) #Carregar o arquivo só com as colunas úteis.

#Ecalonamento é mais interessante por causa dos outliers
#Normalização pode ser interessante também.
from sklearn.preprocessing import MinMaxScaler
#Faz a normalização das outras variáveis
scaler_x = MinMaxScaler()
base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])

#Faz a normalização do preço
scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])

x = base.drop('price', axis = 1) #Atribui todas as colunas menos à que foi passada como parâmetro
y = base.price #Atribui a coluna  variável y

previsores_colunas = colunas_usadas[1:17]
import tensorflow as tf
colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas] #Cria cada coluna no padrão tensorflow

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3) #Divide os dados para treino e teste em uma escala de 70-30

#Cria a função de treino e a de teste.
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento, batch_size = 32, num_epochs = None, shuffle = True)
funcao_teste = tf.estimator.inputs.pandas_input_fn(x = x_teste, y = y_teste, batch_size = 32, num_epochs = 10000, shuffle = False)

#Cria o regressor
regressor = tf.estimator.LinearRegressor(feature_columns = colunas)

#Cria e executa as métricas para analisar o resultado
regressor.train(input_fn=funcao_treinamento, steps = 10000)
metricas_treinamento = regressor.evaluate(input_fn=funcao_treinamento, steps=10000)
metricas_teste = regressor.evaluate(input_fn=funcao_teste, steps=10000)

#Faz as previsões
funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, shuffle = False)
previsoes = regressor.predict(input_fn=funcao_previsao)

#Joga em uma lista somente as colunas previsões
valores_previsoes = []
for p in regressor.predict(input_fn=funcao_previsao):
    valores_previsoes.append(p['predictions'])

#"Desnormaliza os valores previstos"
import numpy as np
valores_previsoes = np.asarray(valores_previsoes).reshape(-1, 1)
valores_previsoes = scaler_y.inverse_transform(valores_previsoes)

#"Desnormaliza os preços originais"
y_teste2 = y_teste.values.reshape(-1, 1)
y_teste2 = scaler_y.inverse_transform(y_teste2)

#Calcula o erro quadrado absoluto
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, valores_previsoes)