import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
raw_data = pd.read_csv('https://raw.githubusercontent.com/stutisehgal/MachineLearning/0d077bf91dcade8ecba67d7a2c0789f48cc15537/Multiple%20Linear%20Regression/chennai_house_multivariate_train.csv')
raw_data.head()
raw_data.describe()
data=(raw_data-raw_data.mean())/(raw_data.max()-raw_data.min())
data.head()
data.min()
data.max()
data.shape
cols = data.shape[1]
print (cols)
data.insert(0, 'Ones', 1)
data.head()
cols = data.shape[1]
print (cols)
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
x = np.matrix(x)
y = np.matrix(y)
params = x.shape[1]
theta = np.matrix(np.array(np.zeros(params)))
x.shape, theta.shape, y.shape
def computeError(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    
    return np.sum(inner) / (2 * len(x))
    computeError(x, y, theta)
Learn_rate=0.15
iters =2000
def gradientDescent(x, y, theta, Learn_rate, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (x * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((Learn_rate/ len(x)) * np.sum(term))
            
        theta = temp
        cost[i] = computeError(x, y, theta)
        
    return theta, cost

new_theta, cost = gradientDescent(x, y, theta, Learn_rate, iters)
print(new_theta, cost)
computeError(x, y, new_theta)
Model_price =  x*new_theta.T

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Iterations')

import math
erro_r = [np.power((b-a),2) for (a, b) in zip(Model_price, y)] #mean absolute percentage error
error0 = np.sum(erro_r)

error=(error0/len(y))*100

print("training error % = {}".format(error))
accuracy= 100 - error
print("training accuracy %={}".format(accuracy))

