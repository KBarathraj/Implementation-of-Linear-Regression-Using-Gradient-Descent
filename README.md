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
\
```

## Output:
### Data Information
![310188551-a7888ba2-5abe-4057-9560-07ba652a980c](https://github.com/gauthamkrishna7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141175025/4015031e-f796-4985-b91e-2bcad67262af)

<br>
<br>
<br>

### Value of X
![310188658-5fde3790-29dc-4ab8-b3ab-a2f431fe3a00](https://github.com/gauthamkrishna7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141175025/ab2f2037-d0d3-4828-ad78-8ca22db3770d)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


### Value of X1_Scaled
![310188895-c5566457-572a-4d4b-adec-af4145ab8d83](https://github.com/gauthamkrishna7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141175025/6a38641e-c102-473b-9bf2-a6afeaabcdc6)

### Predicted Value
![310189062-117a2466-ed22-4985-bd40-849e5ccc5edf](https://github.com/gauthamkrishna7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141175025/ae2ae2a4-16a5-4639-814d-5530c01c616d)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
