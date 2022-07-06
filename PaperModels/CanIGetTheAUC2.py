import matplotlib.pyplot as plt
import xgboost_purple, xgboost_blue, xgboost_brown, xgboost_green, xgboost_orange, xgboost_yellow
import xgboost_model
from sklearn import metrics
import numpy


fpr1, tpr1, threshold1 = metrics.roc_curve(xgboost_model.Y_test, xgboost_model.preds[:,1])
auc1 = metrics.auc(fpr1, tpr1)

fpr2, tpr2, threshold2 = metrics.roc_curve(xgboost_model.purpleTest, xgboost_model.purplePreds[:,1])
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
auc7 = metrics.auc(fpr7, tpr7)

# These are the slices of the model


plt.figure()
lw = 2
plt.plot(
    fpr1,
    tpr1,
    color="black",
    lw=lw,
    label="cumulative model (area = %0.2f)" % auc1,
)
plt.plot(
    fpr2,
    tpr2,
    color="purple",
    lw=lw,
    label="purple model (area = %0.2f)" % auc2,
)

plt.plot(
    fpr3,
    tpr3,
    color="blue",
    lw=lw,
    label="blue model (area = %0.2f)" % auc3,
)

plt.plot(
    fpr4,
    tpr4,
    color="brown",
    lw=lw,
    label="brown model (area = %0.2f)" % auc4,
)

plt.plot(
    fpr5,
    tpr5,
    color="green",
    lw=lw,
    label="green model (area = %0.2f)" % auc5,
)

plt.plot(
    fpr6,
    tpr6,
    color="orange",
    lw=lw,
    label="orange model (area = %0.2f)" % auc6,
)

plt.plot(
    fpr7,
    tpr7,
    color="yellow",
    lw=lw,
    label="yellow model (area = %0.2f)" % auc7,
)

plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic using the model slice splits")
plt.legend(loc="lower right")
plt.show()