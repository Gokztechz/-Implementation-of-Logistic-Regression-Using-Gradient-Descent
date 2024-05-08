# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.Obtain the graph.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: R.GOKUL SHARAN
RegisterNumber:  212223040052
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
*/

*/
```

## Output:
Array Value of x:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/12e7f4e2-94ad-4900-93be-006477415402)
Array Value of y:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/f582dcdc-dee3-452b-be78-63e6a6ae1168)
Exam 1 - score graph:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/5dcb3593-811f-4540-bec1-e21890b75b9a)
Sigmoid function graph:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/b79b211d-0c88-444b-b692-f6062770c610)
X_train_grad value:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/e1d2add1-bfe5-492b-a0c0-9ae63a0202f5)
Y_train_grad value:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/a17bdf06-29a2-4693-886a-24fe06d6dc41)
Print res.x:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/bc2c7ce4-a8ea-488b-98ff-7d96de751fbc)
Decision boundary - graph for exam score:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/d1f9197e-3890-4c52-8901-04431eb8ba0c)
Proability value:
![image](https://github.com/Gokztechz/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117667038/f3693630-629a-40d3-ba75-0251033fb357)
Prediction value of mean:
Prediction value of mean:























![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

