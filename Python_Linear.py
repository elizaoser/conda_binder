#!/usr/bin/env python
# coding: utf-8

# # Python Linear Regression Model

# # Import pandas, sklearn.linear_model, matplotlib.pyplot
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

print(sys.argv)

print("Running linear modelling of data python script")
print()

# # Set notebook variables
filename = "regrex1.csv"

print("Loading filename {}".format(filename))
print()

# # Use the read_csv() function
dataset = pd.read_csv(filename)
dataset.describe
print(dataset)

plt.scatter(dataset[['x']], dataset[['y']], color = 'red')


# # Fitting Linear Regression to the Dataset
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# # Adjusted R-squared
model.score(dataset[['x']], dataset[['y']])


# # Visualizing the Linear Regression results
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


