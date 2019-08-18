from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot = True) #Download da base de dados mnist

#Carregando atributos e classes para treinamentos e testes
x_treinamento = mnist.train.images
y_treinamento = mnist.train.labels
x_teste = mnist.test.images
y_teste = mnist.test.labels

#Biblioteca para exibir graficamente as imagens
import matplotlib.pyplot as plt
import numpy as np
plt.imshow(x_treinamento[0].reshape((28,28)))
plt.title('Classe: ' + str(np.argmax(y_treinamento[0])))

#Dividindo os "blocos" para um treinamento mais otimizado
x_batch, y_batch = mnist.train.next_batch(64)

#784 - 397 - 397 - 397 - 10
#Calculando e atribuindo a quantidade de neurônios de cada camada. Entrada, oculta1, oculta2, oculta3, saida
neuronios_entrada = x_treinamento.shape[1]
neuronios_oculta1 = int((x_treinamento.shape[1] + y_treinamento.shape[1]) / 2)
neuronios_oculta2 = neuronios_oculta1
neuronios_oculta3 = neuronios_oculta1
neuronios_saida = y_treinamento.shape[1]

import tensorflow as tf
#Criando pesos de todas as camadas
w = {'oculta1': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta1])),
     'oculta2': tf.Variable(tf.random_normal([neuronios_oculta1, neuronios_oculta2])),
     'oculta3': tf.Variable(tf.random_normal([neuronios_oculta2, neuronios_oculta3])),
     'saida': tf.Variable(tf.random_normal([neuronios_oculta3, neuronios_saida]))
}

#Criando bias de todas as camadas
b = {
    'oculta1': tf.Variable(tf.random_normal([neuronios_oculta1])),
    'oculta2': tf.Variable(tf.random_normal([neuronios_oculta2])),
    'oculta3': tf.Variable(tf.random_normal([neuronios_oculta3])),
    'saida': tf.Variable(tf.random_normal([neuronios_saida]))
}

#Placeholders
xph = tf.placeholder('float', [None, neuronios_entrada])
yph = tf.placeholder('float', [None, neuronios_saida])

#Função da rede neural
def mlp(x, w, bias):
    camada_oculta1 = tf.nn.relu(tf.add(tf.matmul(x, w['oculta1']), bias['oculta1']))
    camada_oculta2 = tf.nn.relu(tf.add(tf.matmul(camada_oculta1, w['oculta2']), bias['oculta2']))
    camada_oculta3 = tf.nn.relu(tf.add(tf.matmul(camada_oculta2, w['oculta3']), bias['oculta3']))
    camada_saida = tf.add(tf.matmul(camada_oculta3, w['saida']), bias['saida'])
    return camada_saida

#Criando o modelo, calculo do erro, e otimizador
modelo = mlp(xph, w, b)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = modelo, labels = yph))
otimizador = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(erro)

#Criando o modelo de previsões
previsoes = tf.nn.softmax(modelo) #Cria as previsões, no formato 0 e 1. Ex: 00001, 00100, 10000, etc...
previsoes_corretas = tf.equal(tf.argmax(previsoes, 1), tf.argmax(yph, 1)) #Atribui todos os resultados em uma coluna só.
taxa_acerto = tf.reduce_mean(tf.cast(previsoes_corretas, tf.float32)) #Calcula taxa de acerto

with tf.Session() as sess: #Cria sessão tensorflow
    sess.run(tf.global_variables_initializer()) #Inicializa variáveis
    for epoca in range(5000): #Quantidade de épocas
        x_batch, y_batch = mnist.train.next_batch(128) #Atribui as variáveis o tamanho dos batches
        _, custo = sess.run([otimizador, erro], feed_dict= {xph: x_batch, yph: y_batch}) #Executa o modelo
    """
        #Só para printar os resultados em tempo de execução
        if epoca % 100 == 0:
            acc = sess.run([taxa_acerto], feed_dict={xph: x_batch, yph: y_batch})
            print('Época: ' + str((epoca + 1)) + ' erro: ' + str(custo) + ' acc: ' + str(acc))
    print('Treinamento concluído')
    """
    #print(sess.run(previsoes_corretas, feed_dict = {xph: x_teste, yph: y_teste}))
    print(sess.run(taxa_acerto, feed_dict = {xph: x_teste, yph: y_teste})) #Calcula e exibe taxa de acerto