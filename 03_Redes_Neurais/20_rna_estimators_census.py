#Carregando arquivo
import pandas as pd
base = pd.read_csv('dados/census.csv')
base.head()

#Função para converter em 0 e 1 a classe de saída
def converte_classe(rotulo):
    if rotulo == ' >50K':
        return 1
    else:
        return 0

base.income = base.income.apply(converte_classe) #Aplica a função acima

x = base.drop('income', axis = 1) #Pega todas as colunas menos a coluna 'income'
y = base.income #Pega somente a coluna income

#Função para dividir a base de dados em treino e teste.
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

import tensorflow as tf
#Converte os atributos categoricos
workclass = tf.feature_column.categorical_column_with_hash_bucket(key = 'workclass', hash_bucket_size = 100)
education = tf.feature_column.categorical_column_with_hash_bucket(key = 'education', hash_bucket_size = 100)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(key = 'marital-status', hash_bucket_size = 100)
occupation = tf.feature_column.categorical_column_with_hash_bucket(key = 'occupation', hash_bucket_size = 100)
relationship = tf.feature_column.categorical_column_with_hash_bucket(key = 'relationship', hash_bucket_size = 100)
race = tf.feature_column.categorical_column_with_hash_bucket(key = 'race', hash_bucket_size = 100)
country = tf.feature_column.categorical_column_with_hash_bucket(key = 'native-country', hash_bucket_size = 100)

#Converte o atributo categorico, mas em valores específicos
sex = tf.feature_column.categorical_column_with_vocabulary_list(key = 'sex', vocabulary_list=[' Male', ' Female'])

#Padroniza os valores
def padroniza_age(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(38.58)), tf.constant(13.64))

def padroniza_finalweight(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(189778.36)), tf.constant(105549.977))

def padroniza_education(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(10.08)), tf.constant(2.57))

def padroniza_capitalgain(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(1077.64)), tf.constant(7385.29))

def padroniza_capitalloos(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(87.30)), tf.constant(402.96))

def padroniza_hour(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(40.43)), tf.constant(12.34))


#Aplica as funções acima e converte os atributos numéricos
age = tf.feature_column.numeric_column(key = 'age', normalizer_fn = padroniza_age)
final_weight = tf.feature_column.numeric_column(key = 'final-weight', normalizer_fn = padroniza_finalweight)
education_num = tf.feature_column.numeric_column(key = 'education-num', normalizer_fn = padroniza_education)
capital_gain = tf.feature_column.numeric_column(key = 'capital-gain', normalizer_fn = padroniza_capitalgain)
capital_loos = tf.feature_column.numeric_column(key = 'capital-loos', normalizer_fn = padroniza_capitalloos)
hour = tf.feature_column.numeric_column(key = 'hour-per-week', normalizer_fn = padroniza_hour)

#Converte os atributos categóricos 2.0
embedded_workclass = tf.feature_column.embedding_column(workclass, dimension = len(base.workclass.unique()))
embedded_education = tf.feature_column.embedding_column(education, dimension = len(base.education.unique()))
embedded_marital = tf.feature_column.embedding_column(marital_status, dimension = len(base['marital-status'].unique()))
embedded_occupation = tf.feature_column.embedding_column(occupation, dimension = len(base.occupation.unique()))
embedded_relationship = tf.feature_column.embedding_column(relationship, dimension = len(base.relationship.unique()))
embedded_race = tf.feature_column.embedding_column(race, dimension = len(base.race.unique()))
embedded_sex = tf.feature_column.embedding_column(sex, dimension = len(base.sex.unique()))
embedded_country = tf.feature_column.embedding_column(country, dimension = len(base['native-country'].unique()))

#Joga tudo para a lista colunas_rna
colunas_rna = [age, embedded_workclass, final_weight, embedded_education, education_num,
               embedded_marital, embedded_occupation, embedded_relationship, embedded_race, embedded_sex,
               capital_gain, capital_loos, hour, embedded_country]

#Parte de treinamento
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treino, y = y_treino, batch_size = 32, num_epochs = None, shuffle = True)
classificador = tf.estimator.DNNClassifier(hidden_units = [8,8], feature_columns = colunas_rna, n_classes = 2)
classificador.train(input_fn = funcao_treinamento, steps = 10000)

#Parte de testes
funcao_teste = tf.estimator.inputs.pandas_input_fn(x = x_teste, y = y_teste, batch_size = 32, num_epochs = 1, shuffle = False)
classificador.evaluate(input_fn = funcao_teste)
