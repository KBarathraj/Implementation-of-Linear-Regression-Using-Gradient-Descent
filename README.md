# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: BARATHRAJ K
RegisterNumber:  212224230033
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("50_Startups.csv")
X = data.iloc[:, :3].values
y = data.iloc[:, -1].values.reshape(-1, 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xb = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
theta = np.zeros((Xb.shape[1], 1))
learning_rate = 0.1
num_iters = 2000
m = len(y)
for _ in range(num_iters):
    predictions = Xb.dot(theta)
    errors = predictions - y
    gradient = (1/m) * Xb.T.dot(errors)
    theta = theta - learning_rate * gradient
new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_scaled = scaler.transform(new_data)
new_with_bias = np.c_[np.ones((1, 1)), new_scaled]
prediction = new_with_bias.dot(theta)
print(f"Predicted Profit: {float(prediction[0,0]):.2f}")

```

## Output:

<img width="1493" height="1191" alt="image" src="https://github.com/user-attachments/assets/b8e25b11-6886-4903-a4b9-045b587073a5" />

<img width="1497" height="939" alt="image" src="https://github.com/user-attachments/assets/a3745460-2cc4-4872-9d02-3274eacd1f9d" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
