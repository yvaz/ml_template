import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import seaborn as sns
import pandas as pd

def calibration_plot(clf,X_test,y_true):

    y_pred = clf.predict_proba(X_test)[:,1]
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    plt.plot(prob_true,prob_pred)
    
def roc_curve_plot(preds,y):

    fpr, tpr, threshold = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def targets_plot(preds,y):
    
    print('TARGETS PREDS')
    print(preds)
    df = pd.DataFrame({'scores':preds,'targets':y})
    sns.histplot(data=df, x='scores', hue='targets', stat="density", common_norm=False)