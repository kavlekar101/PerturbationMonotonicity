import matplotlib.pyplot as plt
import xgboost_purple, xgboost_blue, xgboost_brown, xgboost_green, xgboost_orange, xgboost_yellow
import xgboost_model
import glmnet_model
from sklearn import metrics
import numpy

# comparing both models 


fpr1, tpr1, threshold1 = metrics.roc_curve(xgboost_model.Y_test, xgboost_model.preds[:,1])
auc1 = metrics.auc(fpr1, tpr1)

fpr2, tpr2, threshold1 = metrics.roc_curve(glmnet_model.Y_test, glmnet_model.fc)
auc2 = metrics.auc(fpr2, tpr2)

'''fpr2, tpr2, threshold2 = metrics.roc_curve(xgboost_model.purpleTest, xgboost_model.purplePreds[:,1])
auc2 = metrics.auc(fpr2, tpr2)

fpr3, tpr3, threshold3 = metrics.roc_curve(xgboost_model.blueTest, xgboost_model.bluePreds[:,1])
auc3 = metrics.auc(fpr3, tpr3)

fpr4, tpr4, threshold4 = metrics.roc_curve(xgboost_model.brownTest, xgboost_model.brownPreds[:,1])
auc4 = metrics.auc(fpr4, tpr4)

fpr5, tpr5, threshold5 = metrics.roc_curve(xgboost_model.greenTest, xgboost_model.greenPreds[:,1])
auc5 = metrics.auc(fpr5, tpr5)

fpr6, tpr6, threshold6 = metrics.roc_curve(xgboost_model.orangeTest, xgboost_model.orangePreds[:,1])
auc6 = metrics.auc(fpr6, tpr6)

fpr7, tpr7, threshold7 = metrics.roc_curve(xgboost_model.yellowTest, xgboost_model.yellowPreds[:,1])
auc7 = metrics.auc(fpr7, tpr7)'''


plt.figure()
lw = 2
plt.plot(
    fpr1,
    tpr1,
    color="black",
    lw=lw,
    label="xgboost model (area = %0.2f)" % auc1,
)
plt.plot(
    fpr2,
    tpr2,
    color="cyan",
    lw=lw,
    label="glmnet model (area = %0.2f)" % auc2,
)

plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic using the model slice splits")
plt.legend(loc="lower right")
plt.show()