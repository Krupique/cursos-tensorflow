"""
    Os placeholders são variáveis do tipo do tipo definido mas são vazias.
    Pode ser alterada em tempo de execução, ela pode ser um número escalar, um vetor, uma matriz, etc...
"""
#Exemplo 1
import tensorflow as tf
p = tf.placeholder('float', None) #Espaço reservado
operacao = p + 2
with tf.Session() as sess:
    resultado = sess.run(operacao, feed_dict={p: [1, 2, 3]})
    print(resultado)

#Exemplo 2
p2 = tf.placeholder('float', [None, 5]) # x linhas e 5 colunas
operacao2 = p2 * 5
with tf.Session() as  sess:
    dados = [[1,2,3,4,5], [6,7,8,9,10]]
    resultado2 = sess.run(operacao2, feed_dict={p2: dados})
    print(resultado2)