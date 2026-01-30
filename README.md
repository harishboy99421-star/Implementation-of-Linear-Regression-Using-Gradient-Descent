# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.

2.Scale the features using a standard scaler to normalize the data.

3.Initialize model parameters (theta) and add an intercept term to the feature set.

4.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data. 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/acer/Downloads/50_Startups
 x=data["R&D Spend"]. values
y=data["Profit"].values

x_mean=np.mean(x)
x_std=np.std(x)
x=(x-x_mean)/x_std

w=0.0
b=0.0
alpha=0.01
epochs=100
n=len(x)

losses=[]

for i in range(epochs):
    y_hat=w*x+b
    loss=np.mean((y_hat-y)**2)
    losses.append(loss)
    dw=(2/n)*np.sum((y_hat-y)*x)
    db=(2/n)*np.sum(y_hat-y)
    w-=alpha*dw
    b-=alpha*db
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(losses)

plt.xlabel("Iteration")

plt.ylabel("Loss (MSE)")

plt.title("Loss vs Iteration")



plt.subplot(1,2,2)

plt.scatter(x,y)

x_sorted=np.argsort(x)

plt.plot(x[x_sorted], (w*x+b)[x_sorted],color='red') 
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")
plt.tight_layout()
plt.show()

print("Final weight(w):",w)
print("Final bias (b):",b)

/*
Program to implement the linear regression using gradient descent.
Developed by: 25019001
RegisterNumber:  Harish.N
*/
```

## Output:

<img width="832" height="385" alt="Screenshot 2026-01-30 193223" src="https://github.com/user-attachments/assets/2087220b-6ec4-4e77-8a7f-55f9fac80610" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
