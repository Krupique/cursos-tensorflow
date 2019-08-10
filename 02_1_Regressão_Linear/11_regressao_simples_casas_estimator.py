import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

#Carregando e exibindo os dados
base = pd.read_csv('dados/house_prices.csv')
base.head()
base.shape

x = base.iloc[:, 5:6].values #Cria uma lista com duas colunas, mas só pega a coluna 5 da base
y = base.iloc[:, 2:3].values

#Escalonamento de x e y
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

colunas = [tf.feature_column.numeric_column('x', shape = [1])]
regressor = tf.estimator.LinearRegressor(feature_columns=colunas)

#Separa os dados em treino e teste em uma escala de 70-30
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size= 0.3)

x_treino.shape
y_treino.shape

x_teste.shape
y_teste.shape

funcao_treinamento = tf.estimator.inputs.numpy_input_fn({'x': x_treino}, y_treino, batch_size = 32, num_epochs = None, shuffle = True) #Retorna uma função que vai alimentar os dados para o tensorflow
funcao_teste = tf.estimator.inputs.numpy_input_fn({'x': x_teste}, y_teste, batch_size = 32, num_epochs = 1000, shuffle = False)
regressor.train(input_fn = funcao_treinamento, steps = 10000)

metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps=10000)
metricas_teste = regressor.evaluate(input_fn = funcao_teste, steps = 10000)

metricas_treinamento
metricas_teste

novas_casas = np.array([[800], [900], [1000]])
novas_casas = scaler_x.transform(novas_casas)

funcao_previsao = tf.estimator.inputs.numpy_input_fn({'x': novas_casas}, shuffle = False)
previsoes = regressor.predict(input_fn = funcao_previsao)

for p in regressor.predict(input_fn = funcao_previsao):
    print(scaler_y.inverse_transform(print(p['predictions'])))