import tensorflow as tf
#Criação da formula para a soma entre dois vetores
a = tf.constant([9, 8, 7], name = 'a')
b = tf.constant([1, 2, 3], name = 'b')
soma = a + b

with tf.Session() as sess:
    print(sess.run(soma))

#Soma de matrizes - exemplo 1
a1 = tf.constant([[1, 2, 3], [4, 5, 6]], name='a1')
b1 = tf.constant([[6, 5, 4], [3, 2, 1]], name='b2')
soma1 = tf.add(a1, b1) #Recurso extra do tensorflow para soma
with tf.Session() as sess:
    print(sess.run(soma1))

#Soma de matrizes - exemplo 2
a2 = tf.constant([[1, 2, 3], [4, 5, 6]], name='a2')
b2 = tf.constant([[1], [2]], name='b2')
soma2 = tf.add(a2, b2)
with tf.Session() as sess:
    print(sess.run(a2))
    print('\n')
    print(sess.run(b2))
    print('\n')
    print(sess.run(soma2))