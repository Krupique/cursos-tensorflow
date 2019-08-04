import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

x = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])
y = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

#Mostra um gráfico com pontinhos
plt.scatter(x, y)
plt.show()

#Objeto da classe LinearRegression
regressor = LinearRegression()
regressor.fit(x, y) #Faz a regressão passando os dois arrays.

#b0
regressor.intercept_
#b1
regressor.coef_

#Previsão manual
previsao1 = regressor.intercept_ + regressor.coef_ * 40
previsao1

#Previsão automática
previsao2 = regressor.predict(40)
previsao2

#Todas as previsões
previsoes = regressor.predict(x)
previsoes

#Calcula a media dos erros absolutos
resultado = abs(y - previsoes).mean()
resultado

#mean absolute error, é mais interassante para fazer a avaliação do algoritmo.
mae = mean_absolute_error(y, previsoes)
#mean square error, é mais interessante quando estiver fazendo o treinamento.
mse = mean_squared_error(y, previsoes)

#Exibe os valores reais e os valores previstos.
plt.plot(x, y, 'o')
plt.plot(x, previsoes, '*', color = 'red')
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Custo')