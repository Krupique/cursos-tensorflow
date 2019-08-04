import tensorflow as tf
entrada = tf.constant([[-1.0, 7.0, 5.0]], name='entradas')
pesos = tf.constant([[0.8, 0.1, 0]], name='pesos')
mult = tf.multiply(entrada, pesos) #Multiplicação escalar
soma = tf.reduce_sum(mult) #Soma de todos os elementos da matriz
with tf.Session() as sess:
    print(sess.run(entrada))
    print(sess.run(pesos))
    print('\n')
    print(sess.run(mult))
    print(sess.run(soma))