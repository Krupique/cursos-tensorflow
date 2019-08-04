import tensorflow as tf
tf.reset_default_graph() #Reseta os grafos anteriores
a = tf.add(2, 2, name='add')
b = tf.multiply(a, 3, name='mult1')
c = tf.multiply(b, a, name='mult2')
"""
    Essa maneira é mais organizada para trabalhar.

with tf.name_scope('Operacoes'):
    with(tf.name_scope('Escopo_A')):
        a = tf.add(2, 2, name='add')
    with(tf.name_scope('Escopo_B')):
        b = tf.multiply(a, 3, name='mult1')
        c = tf.multiply(b, a, name='mult2')
"""
with tf.Session() as sess:
    writer = tf.summary.FileWriter('output', sess.graph)
    print(sess.run(c))
    writer.close()

"""
    Comando para visualizar o gráfico das operações no navegador
    tensorboard --logdir=dir/output --host localhost --port 8088
"""

#Para fazer operações em mais de um grafo, pode ser útil para trabalhar com o conceito de paralelismo
grafo1 = tf.get_default_graph()
grafo1
grafo2 = tf.Graph()
grafo2

with grafo2.as_default():
    print(grafo2 is tf.get_default_graph())