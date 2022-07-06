import numpy as np
from sklearn import linear_model
import pandas as pd

foodData = pd.read_csv("../food_inspec_data.csv", index_col=0, dtype={"Inspector" : "category"})
oneHotInspector = pd.get_dummies(foodData.Inspector, prefix="")
foodData = foodData.drop(["Inspector", "Inspection_ID"], axis=1)
foodData = pd.concat([oneHotInspector, foodData], axis=1)
testData = foodData[foodData["Test"] == True].drop(["Test"], axis=1)
trainData = foodData[foodData["Test"] == False].drop(["Test"], axis=1)
X_train = trainData.iloc[:,:-1]
Y_train = trainData["criticalFound"]
X_test = testData.iloc[:,:-1]
Y_test = testData["criticalFound"]

reg = linear_model.Lasso(normalize = True, alpha = 0.55)
reg.fit(X_train.to_numpy(), Y_train.to_numpy())

print(reg.score(X_train.to_numpy(), Y_train.to_numpy()))
