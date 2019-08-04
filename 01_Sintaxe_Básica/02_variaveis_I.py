x = 35
y = x + 35
print(y)

#O tensorflow só executa um código quando é chamado
#É o chamado computação lazy, que permite melhorias na execução do código.
import tensorflow as tf
valor1 = tf.constant(15, name = 'valor1')
print(valor1)
valor1 = tf.constant(15)
soma = tf.Variable(valor1 + 5, name= 'valor1')
print(soma)
type(soma)
#Definição do grafo de soma acima.

init = tf.global_variables_initializer() #As variáveis tem que ser inicializadas

#Criação de uma sessão para executar a soma
with tf.Session() as sess:
    sess.run(init)
    var = sess.run(soma)