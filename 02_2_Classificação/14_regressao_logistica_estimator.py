#Parte de preprocessamento dos dados
#Carregando o arquivo.
import pandas as pd
base = pd.read_csv('dados/census.csv')
base['income'].unique()
base.head()

#Função para transformar a classe em dados numéricos.
def converte_classe(rotulo):
    if rotulo == ' >50K':
        return 1
    else:
        return 0

#Aplica a função de um jeito diferente e mais pratico.
base['income'] = base['income'].apply(converte_classe)
base.head()
base['income'].unique()

x = base.drop('income', axis = 1) #Adiciona todos os atributos a variável x, menos a class.
y = base['income'] #Adiciona somente a classe a variável y.

base.age.hist() #Mostra um histograma com os dados da coluna 'age'

#Cria um atributo chamado idade para poder transformar a idade em faixas
import  tensorflow as tf
idade = tf.feature_column.numeric_column('age')
idade_categorica = [tf.feature_column.bucketized_column(idade, boundaries=[20,30,40,50,60,70,80,90])]

x.columns #Printa as colunas de x

#Pega somente os dados categóricos
nome_colunas_categoricas = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
colunas_categoricas = [tf.feature_column.categorical_column_with_vocabulary_list(key = c, vocabulary_list = x[c].unique()) for c in nome_colunas_categoricas]
print(colunas_categoricas[2])

#Pega somente os dados numéricos
nome_colunas_numericas = ['final-weight', 'education-num', 'capital-gain', 'capital-loos', 'hour-per-week']
colunas_numericas = [tf.feature_column.numeric_column(key = c) for c in nome_colunas_numericas]
print(colunas_numericas[1])

#Junta todos os grupos acima
colunas = idade_categorica + colunas_categoricas + colunas_numericas

#Para fazer a divisão em treinamento e teste.
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3)

#Parte de treinamento e analise de resultados
#Definição das funções de treinamento e de classificação, respectivamente
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento, batch_size = 32, num_epochs = None, shuffle = True)
classificador = tf.estimator.LinearClassifier(feature_columns = colunas)

#Treinamento (Pode demorar algum tempinho para executar)
classificador.train(input_fn = funcao_treinamento, steps = 10000)

#Definição da função de previsão
funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, batch_size = 32, shuffle = False)
#Executar previsão
previsoes = classificador.predict(input_fn = funcao_previsao)

#Visualizar
list(previsoes)

#Separar os resultados em uma lista "comum"
previsoes_final = []
for p in classificador.predict(input_fn = funcao_previsao):
    previsoes_final.append(p['class_ids'])

#Visualizar taxa de acerto
from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste, previsoes_final)
taxa_acerto