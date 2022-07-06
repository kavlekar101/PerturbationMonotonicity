import xgboost
import pandas as pd
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
foodData = pd.read_csv("food_inspec_data.csv", index_col=0, dtype={"Inspector" : "category"})
foodData = foodData[foodData['Inspector'] == "purple"] # gets only the purple inspectors


print(foodData.shape[0])


oneHotInspector = pd.get_dummies(foodData.Inspector, prefix="")
foodData = foodData.drop(["Inspector", "Inspection_ID"], axis=1)
foodData = pd.concat([oneHotInspector, foodData], axis=1)
testData = foodData[foodData["Test"] == True].drop(["Test"], axis=1)
trainData = foodData[foodData["Test"] == False].drop(["Test"], axis=1)

X_train = trainData.iloc[:,:-1]
Y_train = trainData["criticalFound"]
X_test = testData.iloc[:,:-1]
Y_test = testData["criticalFound"]

dtrain = xgboost.DMatrix(X_train, label=Y_train)
dtest = xgboost.DMatrix(X_test, label=Y_test)

params = {
    'objective': "multi:softprob",
    'gamma': 0,
    'max_depth': 6,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'eta': 0.02,
    'nthread': 4,
    "num_class":2,
    "eval_metric":"mlogloss",
    "num_parallel_tree": 500,
}

num_round = 2  # the number of training iterations. It takes too much time on my m1 chip for 500 iters.
bstmodel = xgboost.train(params, dtrain, num_round)

#Save as human readable model
# bstmodel.dump_model('dump.raw.txt')
preds = bstmodel.predict(dtest)
#print(preds)

import numpy as np
print(np.average(preds, 0))

# print(bstmodel.get_score(importance_type="gain"))

# preds.shape
# preds

from sklearn import metrics
mse = metrics.mean_squared_error(Y_test, preds[:,-1])

#print('mean square error: %f' % mse)
