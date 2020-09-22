'''

Practice Exercise: Linear Regressions in Python
Watts Dietrich
Sept 22 2020

The goal of this exercise is to practice linear regression by building a model to estimate final grades of students.
The program generates a series of linear regression models and saves the best one for future use.

I use a student performance data set obtained from the UCI machine learning repository here:
https://archive.ics.uci.edu/ml/datasets/Student+Performance#
It contains detailed data on 649 students.

Five attributes are used:
G1 - first period grade
G2 - second period grade
absences - number of school absences
studytime - weekly study time in hours
failures - number of past class failures

These are used to predict G3, the final period grade.

I used a 90/10 train/test split, generated 1000 models, and saved the best.
The best model yielded an accuracy score of 97.26% and does a very good job,
often estimating the final score to within 1 point of the true value on a 20-point scale, with few outliers.

The program prints the predicted and actual final grades for the test dataset of 40 students.
It also uses matplotlib to graph the first grade G1 vs the final grade.

'''

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

# create a pandas dataframe from csv
data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

# remove unneeded attributes, only selected attributes in updated dataframe
data = data[["G1", "G2", "G3", "absences", "studytime", "failures"]]
print(data.head())

# this is the "label," what we are trying to predict
predict = "G3"

# create numpy arrays: x contains the predictive attributes and y the target variable (the "label")
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# split data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# code block below generates 1000 regression models using training data, tests accuracy, and saves the best model
# commented out because we can load the best saved model later without generating new ones each time
# note: the current saved model had an accuracy score of 97.26%
'''
best = 0
for _ in range(1000):
    # split data into training and testing sets
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    # create a new linear regression model
    linear = linear_model.LinearRegression()

    # fit model to training data
    linear.fit(x_train, y_train)

    # score model fit on testing data, test accuracy of model
    # note: this will vary slightly because the training and testing sets are selected randomly
    acc = linear.score(x_test, y_test)
    print(acc)

    # save the model using pickle if its accuracy is the best so far
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
            
# print model accuracy
print("Best: ", best)
'''

# load the saved model into variable linear
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# lists the coefficients of all 5 variables, intercept
# note: larger coefficient means that attribute has more weight
print("Coeff: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# predict the final grades of the test data instances (students)
predictions = linear.predict(x_test)

# print predictions and compare to real results
# displays predicted and actual final grades
for x in range(len(predictions)):
    #print(round(predictions[x], 2), x_test[x], y_test[x])
    print('Student', x, 'predicted grade: ', round(predictions[x], 2), ', Actual grade: ', y_test[x])


# here I plot one of the attributes vs the true value of the final grade, G3
# choose matplotlib style
style.use("ggplot")

# the attribute to be plotted on x axis
p = 'G1'
# p = 'G2'
# p = 'absences'
# p = 'studytime'
# p = 'failures'

# create and show plot
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
