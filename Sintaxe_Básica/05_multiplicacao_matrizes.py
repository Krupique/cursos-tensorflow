#Exemplo 1
import tensorflow as tf
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[-1, 3], [4, 2]])
mult1 = tf.matmul(a, b) #Operação para fazer a multiplicação entre matrizes
mult2 = tf.matmul(b, a) #Operação para fazer a multiplicação entre matrizes
with tf.Session() as sess:
    print(sess.run(a))
    print('\n')
    print(sess.run(b))
    print('\n')
    print(sess.run(mult1))
    print('\n')
    print(sess.run(mult2))

#Exemplo 2
a1 = tf.constant([[2,3],[0,1],[-1,4]])
b1 = tf.constant([[1, 2,3],[-2, 0, 4]])
mult3 = tf.matmul(a1, b1)
with tf.Session() as sess:
    print(sess.run(mult3))