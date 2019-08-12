#Carregando os arquivos.
import pandas as pd
base = pd.read_csv('dados/census.csv')
base.head()
base['income'].unique() #Unique serve para mostrar todos os valores sem repetir.
base.shape

#Dividindo os atributos das classes
x = base.iloc[:, 0:14].values
y = base.iloc[:, 14].values

#Label encoder serve para transformar os dados categoricos em dados numéricos
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

#Percisa fazer a transformação para cada atributo individualmente.
x[:, 1] = label_encoder.fit_transform(x[:, 1])
x[:, 3] = label_encoder.fit_transform(x[:, 3])
x[:, 5] = label_encoder.fit_transform(x[:, 5])
x[:, 6] = label_encoder.fit_transform(x[:, 6])
x[:, 7] = label_encoder.fit_transform(x[:, 7])
x[:, 8] = label_encoder.fit_transform(x[:, 8])
x[:, 9] = label_encoder.fit_transform(x[:, 9])
x[:, 13] = label_encoder.fit_transform(x[:, 13])

#Fazendo o escalonamento dos atributos.
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

#Fazendo a divisão entre treinamento e teste em uma escala de 70 - 30.
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3)

#Fazendo o modelo
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(max_iter = 10000)
classificador.fit(x_treinamento, y_treinamento)

#Fazendo as previsões
previsoes = classificador.predict(x_teste)

#Calculando a taxa de acerto
from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste, previsoes)
