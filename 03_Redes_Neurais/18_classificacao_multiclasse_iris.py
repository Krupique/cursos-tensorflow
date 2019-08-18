from sklearn import datasets
iris = datasets.load_iris() #Dataset iris, o das plantas

x = iris.data #Colunas com os atributas
y = iris.target #Coluna com as clases

#Transforma todos os atributos em valores escalonados
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

#Transforma todas classes em escala, porém utilizando 1 e 0. Ex: 3 classes vai ficar 100, ou 010, ou 001.
#Isso se chama OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features=[0])
y = y.reshape(-1, 1)
y = onehot.fit_transform(y).toarray()

#Divide o conjunto em treino e teste em uma proporção de 70-30
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3)

import tensorflow as tf
import numpy as np

neuronios_entrada = x.shape[1] #Quantidade de neurônios de entrada, dado pela quantidade de atributos
neuronios_oculta = int(np.ceil((x.shape[1] + y.shape[1]) / 2)) #Quantidade de neurônios da camada oculta, dado pela média aritmética da quantidade de neurônios de entrada e de saída
neuronios_saida = y.shape[1] #Quantidade de neurônios de saída, dado pela quantidade de classes

#Inicialização dos pesos e dos bias das camadas ocultas, e de saídas.
w = {'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta])),
     'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]))}
b = {'oculta': tf.Variable(tf.random_normal([neuronios_oculta])),
     'saida': tf.Variable(tf.random_normal([neuronios_saida]))}
#Declaração dos placeholders para armazenar os dados que serão submetidos ao tensorflow
xph = tf.placeholder('float', [None, neuronios_entrada])
yph = tf.placeholder('float', [None, neuronios_saida])

#Função para gerar o modelo mlp
def mlp(x, w, bias):
    camada_oculta = tf.add(tf.matmul(x, w['oculta']), bias['oculta'])
    camada_oculta_ativacao = tf.nn.relu(camada_oculta)
    camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, w['saida']), b['saida'])
    return camada_saida

#Cálculo do modelo, do erro e otimizador para a rede """Ainda não entendo direito essa parte"""
modelo = mlp(xph, w, b)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = modelo, labels = yph))
otimizador = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(erro)

#Inicialização dos batchs, servem para dividir as entradas em "blocos"
batch_size = 8
batch_total = int(len(x_treinamento) / batch_size)
x_batches = np.array_split(x_treinamento, batch_total)

#Sessão tensorflow para fazer o treinamento.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(3000): #Vai fazer 3000 épocas
        erro_medio = 0.0
        batch_total = int(len(x_treinamento) / batch_size)
        x_batches = np.array_split(x_treinamento, batch_total)
        y_batches = np.array_split(y_treinamento, batch_total)
        for i in range(batch_total):
            x_batch, y_batch = x_batches[i], y_batches[i]
            _, custo = sess.run([otimizador, erro], feed_dict = {xph: x_batch, yph: y_batch})
            erro_medio += custo / batch_total
        #if epoca % 500 == 0:
        #    print('Época: ' + str((epoca + 1)) + ' erro: ' + str(erro_medio))
    w_final, b_final = sess.run([w, b])

#Previsões
previsoes_teste = mlp(xph, w_final, b_final) #Jogar os pesos e o bias no modelo
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r1 = sess.run(previsoes_teste, feed_dict= {xph: x_teste}) #Vai fazer as previsões baseadas no modelo
    r2 = sess.run(tf.nn.softmax(r1)) #Transformar os dados resultantes em probabilidades
    r3 = sess.run(tf.argmax(r2, 1)) #Escolhe o maior valor dentre os resultantes e seta 1, enquanto que os outros recebem 0

y_teste2 = np.argmax(y_teste, 1) #Transforma os resultados em números baseado nas suas respectivas posições da tabela.

#Calcular a taxa de acerto (de 0 à 1)
from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste2, r3)
