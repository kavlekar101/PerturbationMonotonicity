import numpy
import pandas as pd
import scipy
import importlib
import pprint
import matplotlib.pyplot as plt
import warnings
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

def Perturbation(df, inspector, frac):
    groupData = df[df[inspector] == 1]
    groupData = groupData[groupData["criticalFound"] == 0].sample(frac=frac, random_state=42)
    for index, row in groupData.iterrows():
        df.at[index, "criticalFound"] = 1
    return df


def criticalDistByIns(test, pred):
    df = pd.DataFrame(pred, columns=["Pred"])
    df.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    test = pd.concat([test, df], axis=1)
    Inspectors = ["_blue", "_brown", "_green", "_orange", "_purple", "_yellow"]
    for inspector in Inspectors:
        all = test[test[inspector] == 1]
        print("Inspector:" + inspector)
        print("Fail rate Avg: " , numpy.mean(all["Pred"].to_numpy()))
        print("std: " , numpy.std(all["Pred"].to_numpy()))


#load all data we need
foodDataOrigin = pd.read_csv("food_inspec_data.csv", index_col=0, dtype={"Inspector" : "category"})
oneHotInspector = pd.get_dummies(foodDataOrigin.Inspector, prefix="")
foodData = foodDataOrigin.drop(["Inspector", "Inspection_ID"], axis=1)
foodData = pd.concat([oneHotInspector, foodData], axis=1)
testData = foodData[foodData["Test"] == True].drop(["Test"], axis=1)
trainData = foodData[foodData["Test"] == False].drop(["Test"], axis=1)
# trainData = Perturbation(trainData, "_purple", 0.1)
X_train = trainData.iloc[:,:-1]
Y_train = trainData["criticalFound"]
X_test = testData.iloc[:, :-1]
Y_test = testData["criticalFound"]

# add penalty, only inspector coef are penalized
penalty = numpy.concatenate((numpy.ones(6),numpy.zeros(10)))

# train the model and print the coef
cvfit = cvglmnet(x = X_train.to_numpy().copy(), y = Y_train.to_numpy().copy(), family = 'binomial', alpha = 0, penalty_factor = penalty)
coef = cvglmnetCoef(cvfit, s = 'lambda_min')
#print(cvfit["lambda_min"])
#print(coef)

# predict the test data
# allData = pd.concat([trainData, testData], axis=0)
# X_test = allData.iloc[:,:-1]


fc = cvglmnetPredict(cvfit, X_test.to_numpy().copy(), ptype = "response", s = cvfit["lambda_min"])
#print(fc.shape)
#print(fc)
#print(type(fc))

#criticalDistByIns(X_test, fc)


