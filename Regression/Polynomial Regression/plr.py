# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting the Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)


#Fittig the polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 7)
X_poly = polyreg.fit_transform(X)

linreg2 = LinearRegression()
linreg2.fit(X_poly, y)

#Visualising the polymomial reg
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linreg2.predict(polyreg.fit_transform(X_grid)))
plt.show()







