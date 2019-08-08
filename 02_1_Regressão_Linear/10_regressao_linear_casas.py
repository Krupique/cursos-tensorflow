import pandas as pd #Para carregamento dos dados
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

base = pd.read_csv('dados/house_prices.csv')
base.head()
base.count()
base.shape #Mostra quantas linhas e quantas colunas tem.

x = base.iloc[:, 5].values #Quero uma lista com todas as linhas e somente a coluna 5. O .values e para transformar em numpy array
x = x.reshape(-1, 1) #Não quero mexer nas linhas, mas quero adicionar uma coluna
x.shape

y = base.iloc[:, 2:3].values #A mesma coisa anterior, mas agora não precisa do reshape por causa do 2:3
y.shape

#Gera warning, pois está convertendo int para float
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

#Fórmula da regressão linear simples: y = b0 + b1 * x
np.random.seed(1)
np.random.rand(2)

b0 = tf.Variable(0.41)
b1 = tf.Variable(0.72)

batch_size = 32 #Pega 32 registros para a base de dados e manda para o algoritmo treinar, depois mais 32 e assim sucessivamente
xph = tf.placeholder(tf.float32, [batch_size, 1]) #O placeholder é uma variável que não vai receber os dados ainda.
yph = tf.placeholder(tf.float32, [batch_size, 1])

y_modelo = b0 + b1 * xph #Modelo com a fórmula
erro = tf.losses.mean_squared_error(yph, y_modelo) #Cálculo do menor erro quadrado
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001) #Otimizador utilizando a descida do gradiente.
treinamento = otimizador.minimize(erro) #Minimizador do otimizador.
init = tf.global_variables_initializer() #Inicializador de variáveis.

#Criando uma sessão.
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        indices = np.random.randint(len(x), size = batch_size) #Pega em uma lista valores aleatórios de x do tamanho do batch_size
        feed = {xph: x[indices], yph: y[indices]}
        sess.run(treinamento, feed_dict = feed)
    b0_final, b1_final = sess.run([b0, b1])


previsoes = b0_final + b1_final * x #Previsões

#Gráfico com o x es previsões
plt.plot(x, y, 'o')
plt.plot(x, previsoes, color ='red')

#"Desescalonando os valores"
y1 = scaler_y.inverse_transform(y)
previsoes1 = scaler_y.inverse_transform(previsoes)

#Cálculo do mínimo quadrado absoluto
mae = mean_absolute_error(y1, previsoes1)