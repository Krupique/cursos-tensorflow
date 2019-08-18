import tensorflow as tf
import numpy as np

x = np.array([[0,0], [0,1], [1,0], [1,1]]) #Matriz de entrada
y = np.array([[1], [0], [0], [1]]) #Matriz de saída

neuronios_entrada = 2 #Quantidade de neurônios de entrada
neuronios_oculta = 3 #Quantidade de neurônios da camada oculta
neuronios_saida = 1 #Quantidade de neurônios da camada de saída

#Pesos da camada oculta e camada de saída
w = {'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta]), name = 'w_oculta'),
     'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]), name= 'w_saida')}

#Bias da camada oculta e camada de saída
b = {'oculta': tf.Variable(tf.random_normal([neuronios_oculta]), name='b_oculta'),
     'saida': tf.Variable(tf.random_normal([neuronios_saida]), name='b_saida')}

distribuicao = np.random.normal(size = 500) #Gerando números aleatórios
#distribuicao
import seaborn as sns #Biblioteca bastante utilizada para visualização de dados
sns.distplot(distribuicao) #Exibir dados

xph = tf.placeholder(tf.float32, [4, neuronios_entrada], name = 'xph') #Armazenando os valores de entrada?
yph = tf.placeholder(tf.float32, [4, neuronios_saida], name = 'yph') #Armazenando os valores de saída?
camada_oculta = tf.add(tf.matmul(xph, w['oculta']), b['oculta']) #Cálculos da camada oculta, multiplicação das matrizes e soma com o bias
camada_oculta_ativacao = tf.sigmoid(camada_oculta) #Aplicação da função de ativação na camada oculta
camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, w['saida']), b['saida']) #Cálculos da camada de saída, multiplicação das matrizes e soma com o bias
camada_saida_ativacao = tf.sigmoid(camada_saida) #Aplicação da função de ativação na camada de saída
erro = tf.losses.mean_squared_error(yph, camada_saida_ativacao) #Cálculo do erro
otimizador = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(erro) #Calculando a descida do gradiente no erro.

init = tf.global_variables_initializer() #Inicializando variáveis
#Criando sessão tensorflow
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(w['oculta']))
    #print(sess.run(w['saida']))
    #print(sess.run(camada_oculta_ativacao, feed_dict= {xph: x}))
    #print(sess.run(camada_saida, feed_dict= {xph: x}))
    for epocas in range(10000): #Treinamento
        erro_medio = 0
        _, custo = sess.run([otimizador, erro], feed_dict= {xph: x, yph: y})
        if epocas % 200 == 0:
            erro_medio += custo/4
            print(erro_medio)
    w_final, b_final = sess.run([w, b])

#Testes
camada_oculta_teste = tf.add(tf.matmul(xph, w_final['oculta']), b_final['oculta']) #Cálculos da camada oculta, multiplicação das matrizes e soma com o bias
camada_oculta_ativacao_teste = tf.sigmoid(camada_oculta_teste) #Aplicação da função de ativação na camada oculta
camada_saida_teste = tf.add(tf.matmul(camada_oculta_ativacao_teste, w_final['saida']), b_final['saida']) #Cálculos da camada de saída, multiplicação das matrizes e soma com o bias
camada_saida_ativacao_teste = tf.sigmoid(camada_saida_teste) #Aplicação da função de ativação na camada de saída
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(camada_saida_ativacao_teste, feed_dict = {xph: x}))