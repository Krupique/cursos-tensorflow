#Exemplo de soma escalar em um vetor
import tensorflow as tf
vetor = tf.constant([5, 10, 15], name='vetor')
type(vetor)
print(vetor)
soma = tf.Variable(vetor + 5, name='soma')
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(soma))

#Exemplo de loop em uma soma
valor = tf.Variable(0, name='valor')
init2 = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init2)
    for i in range(5):
        valor = valor + 1
        print(sess.run(valor))