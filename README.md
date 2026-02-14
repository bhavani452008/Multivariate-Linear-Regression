# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd.

### Step2
Read the csv file.

### Step3
Get the value of X and y variables

### Step4
Create the linear regression model and fit.

### Step5
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)


reg = linear_model.LinearRegression()


reg.fit(X_train, y_train)

print('Coefficients: \n', reg.coef_)

print('Variance score: {}'.format(reg.score(X_test, y_test)))

plt.style.use('fivethirtyeight')


plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')


plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')


plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()
```
## Output:

### Insert your output
<img width="419" height="843" alt="image" src="https://github.com/user-attachments/assets/52d95c1a-3eea-4f55-9659-f59ced37b2d4" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
