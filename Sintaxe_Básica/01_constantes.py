import tensorflow as tf
valor1 = tf.constant(2)
valor2 = tf.constant(3)
type(valor1)
print(valor1)

soma = valor1 + valor2
type(soma)
print(soma)

#A soma foi definida mas não foi executada, para executar é preciso criar uma sessão.
with tf.Session() as sess:
    s = sess.run(soma)
    print(s)

#Independente se a variável é int, string ou outro aqui será sempre do tipo tensor
text1 = tf.constant('Texto 1')
text2 = tf.constant('Texto 2')
type(text1)
type(text2)
print(text1)

with tf.Session() as sess:
    con = sess.run(text1 + text2)

print(con)