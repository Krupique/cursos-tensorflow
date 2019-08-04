import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf

x = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])
y = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

#Quando trabalhamos com Tensorflow precisamos fazer o escalonamento dos valores.
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

plt.scatter(x, y)
#Fórmula da regressõ linear simples: y = b0 + b1 * x
np.random.seed(0)
np.random.rand(2)

#Pesos iniciais
b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)

erro = tf.losses.mean_squared_error(y, (b0 + b1 * x)) #Faz a fórmula
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001) #Treina a descida do gradiente
treinamento = otimizador.minimize(erro) #Minimização do erro
init = tf.global_variables_initializer() #Inicializa as variáveis

#Sessão tensorflow para fazer o treinamento da rede
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(b0))
    #print(sess.run(b1))
    for i in range(1000):
        sess.run(treinamento)
    b0_final, b1_final = sess.run([b0, b1])

previsoes = b0_final + b1_final * x #Calculo das previsões pela fórmula

#Exibindo as previsões e os valores reais
plt.plot(x, previsoes, color = 'green')
plt.plot(x, y, 'o')


scaler_x.transform([[40]]) #Escalona o número 40.
#Faz o inverso do escalonamento e faz a fórmula para prever com um número "real"
previsao = scaler_y.inverse_transform(b0_final + b1_final * scaler_x.transform([[40]]))

#Faz o inverso do escalonamento de y e previsões
y1 = scaler_y.inverse_transform(y)
previsoes1 = scaler_y.inverse_transform(previsoes)

mae = mean_absolute_error(y1, previsoes1) #Erros absolutos
mse = mean_squared_error(y1, previsoes1) #Erros quadraticos
