import tensorflow as tf
import numpy as np

#Entrada
x = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])

#Saida
y = np.array([[0.], [0.], [0.], [1.]])

#Pesos, inicializando
w = tf.Variable(tf.zeros([2 , 1], dtype=tf.float64))

#Inicializando variável para o tensorflow
init = tf.global_variables_initializer()
camada_saida = tf.matmul(x, w) #Multiplicação de matrizes da entrada pelos pesos
camada_saida_ativao = step(camada_saida) #Aplicando a função de ativação a saída do somatório

#Função de ativação
def step(x):
    return tf.cast(tf.to_float(tf.math.greater_equal(x, 1)), tf.float64)

erro = tf.subtract(y, camada_saida_ativao) #Calculando o erro.

delta = tf.matmul(x, erro, transpose_a = True) #Multiplicação da transposta de 'x' por 'erro'
treinamento = tf.assign(w, tf.add(w, tf.multiply(delta, 0.1))) #Faz o resto

#Sessão tensorflow
with tf.Session() as sess:
    sess.run(init) #Inicializando variáveis
    """
    print(sess.run(camada_saida)) #Rodando a primeira parte
    print('\n')
    print(sess.run(camada_saida_ativao)) #Rodando a função de ativação
    print('\n')
    print(sess.run(erro)) #Rodando o cálculo do erro
    """
    epoca = 0
    for i in range(15): #Roda no máximo 15 épocas
        epoca += 1
        erro_total, _ = sess.run([erro, treinamento])
        erro_soma = tf.reduce_sum(erro_total)
        print('Epoca: ', epoca, ' Erro: ', sess.run(erro_soma))
        if erro_soma.eval() == 0.0: #Se o somatório do erro for 0, então para de fazer
            break
    w_final = sess.run(w)

#Teste
camda_saida_teste = tf.matmul(x, w_final)
camada_saida_ativacao_teste = step(camda_saida_teste)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(camada_saida_ativacao_teste))