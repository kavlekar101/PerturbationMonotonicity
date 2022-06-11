import numpy
import pandas as pd
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
import glmnet_python
from glmnet import glmnet
from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint
from glmnetCoef import glmnetCoef
from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot
from cvglmnetPredict import cvglmnetPredict
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
foodData = pd.read_csv("food_inspec_data.csv", index_col=0, dtype={"Inspector" : "category"})
oneHotInspector = pd.get_dummies(foodData.Inspector, prefix="")
foodData = foodData.drop(["Inspector", "Inspection_ID"], axis=1)
foodData = pd.concat([oneHotInspector, foodData], axis=1)
testData = foodData[foodData["Test"] == True].drop(["Test"], axis=1)
trainData = foodData[foodData["Test"] == False].drop(["Test"], axis=1)
X_train = trainData.iloc[:,:-1]
Y_train = trainData["criticalFound"]
X_test = testData.iloc[:,:-1]
Y_test = testData["criticalFound"]

penalty = numpy.concatenate((numpy.ones(6),numpy.zeros(10)))
print(penalty)

cvfit = cvglmnet(x = X_train.to_numpy().copy(), y = Y_train.to_numpy().copy(), family = 'binomial', alpha = 0, penalty_factor = penalty)
coef = cvglmnetCoef(cvfit, s = 'lambda_min')
print(cvfit["lambda_min"])
print(coef)
