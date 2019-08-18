import numpy as np

#Utilizada somente para problemas linearmente separáveis
def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0

#Sigmoide, o principal uso é para fazer a classificação em problemas binários.
def sigmoideFunction(soma):
    return 1 / (1 + np.exp(-soma))

#Tangente hiperbólica (Retorna valores entre -1 e 1)
def tahnFunction(soma):
    return (np.exp(soma) / np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

#ReLU (rectified linear units) Retorna valores >= 0 sem limite max. É bastante usada em CNNs e Deep Learning
def relu(soma):
    if soma >= 0:
        return soma
    return 0

#Linear. Retorna o próprio valor, é bastante utilizada em regressão.
def linearFunction(soma):
    return soma

#Softmax. É bastante utilizada em Deep Learning com mais de duas classes de saída.
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

#Para saber mais, acessar o site da documentação do Keras.

print(tahnFunction(2.1))