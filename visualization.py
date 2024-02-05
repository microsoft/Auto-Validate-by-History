from utils import *
from tfdv import *
from deequ import *
from azureAD import *
from azure_drift_detector import *
import avh_no_stationary as k_v3
import avh_with_stationary as k_v4
from hypothesis_test import *
from ml_baseline_single_var import *


def plot_numeric():
    """
    load and plot result
    """
    
    # Google TFDV
    tfdv = TFDV(is_numeric=True)
    tfdv_precision_real, tfdv_precision_syn, tfdv_recall_real, tfdv_recall_syn = tfdv.parse_result_update()

    # Amazon Deequ
    deequ = Deequ()
    deequ_precision_real, deequ_precision_syn, deequ_recall_real, deequ_recall_syn = deequ.parse_result_update(is_numeric=True)

    # Azure Anomaly Detection
    azure_ad = AzureAD(True)
    azure_precision, azure_recall = azure_ad.parse_result_update()

    # Azure Drift Detection
    azure_drift_detector = AzureDriftDetector(True)
    add_precision_real,add_precision_syn, add_recall_real, add_recall_syn = azure_drift_detector.parse_result_update()

    # AVH_with_stationary
    k_clause = k_v4.KClause(is_numeric=True)
    k_clause.save_base_dir = '../result/k_clause_v4/numeric_result'
    k_v4_precision_real, k_v4_precision_syn, k_v4_recall_real, k_v4_recall_syn = k_clause.parse_result_update()

    # Hypothesis testing
    hypo = HypoTest(is_numeric=True, verbose=True)
    hypo_precision_real, hypo_precision_syn, hypo_recall_real, hypo_recall_syn = hypo.parse_result_update()

    # One-Class SVM
    ad = AnomalyDetection(is_numeric=True, model_name='svm')
    svm_precision_real, svm_precision_syn, svm_recall_real, svm_recall_syn = ad.parse_result_update()

    # Isolation Forest
    ad = AnomalyDetection(is_numeric=True, model_name='isof')
    isof_precision_real, isof_precision_syn, isof_recall_real, isof_recall_syn = ad.parse_result_update()

    # LocalOutlierFactor
    ad = AnomalyDetection(is_numeric=True, model_name='lof')
    lof_precision_real, lof_precision_syn, lof_recall_real, lof_recall_syn = ad.parse_result_update()

    # XGboost
    sm = SupervisedModel(is_numeric=True, model_name='xgb')
    xgb_precision_real, xgb_precision_syn, xgb_recall_real, xgb_recall_syn = sm.parse_result_update()



    ################### Test on real data ########################
    plt.figure(figsize=(15, 10), dpi=80)

    # AVH
    plt.plot(k_v4_recall_real, k_v4_precision_real, linestyle='solid', linewidth=5, label="AVH", marker='*', markersize=12, c='r')
    plt.annotate("AVH", 
                xy=(k_v4_recall_real[8], k_v4_precision_real[8]), 
                xytext=(k_v4_recall_real[8]+0.02,  k_v4_precision_real[8]+0.035),
                arrowprops=dict(facecolor="r", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # # Azure Anomaly Detector
    plt.plot(azure_recall[1:], azure_precision[1:], linestyle='dashdot', linewidth=5, label="Azure Anomaly Detector", marker='o', markersize=10, c='g')
    plt.annotate("Azure Anomaly Detector", 
                xy=(azure_recall[-3], azure_precision[-3]), 
                xytext=(azure_recall[-3]-0.16, azure_precision[-3]+0.036),
                arrowprops=dict(facecolor="g", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # azure drift detector(add)
    plt.plot(add_recall_real[1:8], add_precision_real[1:8], linestyle='dashdot', linewidth=5, label="Azure Drift Detector", marker='s', markersize=10, c='darkorange')
    plt.annotate("Azure Drift Detector", 
                xy=(add_recall_real[7], add_precision_real[7]), 
                xytext=(add_recall_real[7]-0.12,  add_precision_real[7]-0.04),
                arrowprops=dict(facecolor="darkorange", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # # TFDV
    plt.plot(tfdv_recall_real[1:14], tfdv_precision_real[1:14], linestyle='dashed', linewidth=5, label="Google TFDV", marker='v', markersize=10, c='dodgerblue')
    plt.annotate("Google TFDV", 
                xy=(tfdv_recall_real[2], tfdv_precision_real[2]), 
                xytext=(tfdv_recall_real[2]-0.18, tfdv_precision_real[2]-0.05),
                arrowprops=dict(facecolor="dodgerblue", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # # Amazon Deequ
    plt.plot(deequ_recall_real, deequ_precision_real, linestyle='dotted', linewidth=5, label="Amazon Deequ", marker='o', markersize=10, c='indigo')
    plt.annotate("Amazon Deequ", 
                xy=(deequ_recall_real[-1], deequ_precision_real[-1]), 
                xytext=(deequ_recall_real[-1]+0.05, deequ_precision_real[-1]-0.04),
                arrowprops=dict(facecolor="indigo", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # # Hypothesis Testing
    plt.plot(hypo_recall_real[1:20][::3], hypo_precision_real[1:20][::3], linestyle='dotted', linewidth=5, label="KS-Test", marker='^', markersize=10, c='brown')
    plt.annotate("KS-Test", 
                xy=(hypo_recall_real[1:20][::3][5], hypo_precision_real[1:20][::3][5]), 
                xytext=(hypo_recall_real[1:20][::3][5]+0.025,  hypo_precision_real[1:20][::3][5]+0.035),
                arrowprops=dict(facecolor="brown", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # # XGB
    plt.plot(xgb_recall_real[100:200:5][11:19], xgb_precision_real[100:200:5][11:19], linestyle='dashed', linewidth=5, label="XGBoost", marker='s', markersize=10, c='pink')
    plt.annotate("XGBoost", 
                xy=(xgb_recall_real[100:200:5][11:19][-1], xgb_precision_real[100:200:5][11:19][-1]), 
                xytext=(xgb_recall_real[100:200:5][11:19][-1]-0.14,  xgb_precision_real[100:200:5][11:19][-1]-0.05),
                arrowprops=dict(facecolor="pink", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # # Anomaly Detection
    plt.plot(svm_recall_real, svm_precision_real, linestyle='dotted', linewidth=3, label="OneClassSVM", marker='v', markersize=10, c='olive')
    plt.annotate("OneClassSVM", 
                xy=(svm_recall_real[1], svm_precision_real[1]), 
                xytext=(svm_recall_real[1]+0.02,  svm_precision_real[1]+0.035),
                arrowprops=dict(facecolor="olive", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    plt.plot(isof_recall_real, isof_precision_real, linestyle='dashdot', linewidth=5, label="Isolaton Forest", marker='*', markersize=10, c='orangered')
    plt.annotate("Isolaton Forest", 
                xy=(isof_recall_real[3], isof_precision_real[3]), 
                xytext=(isof_recall_real[3]-0.2,  isof_precision_real[3]-0.05),
                arrowprops=dict(facecolor="orangered", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    plt.plot(lof_recall_real, lof_precision_real, linestyle='dashed', linewidth=5, label="LocalOutlierFactor", marker='s', markersize=10, c='purple')
    plt.annotate("LocalOutlierFactor", 
                xy=(lof_recall_real[1], lof_precision_real[1]), 
                xytext=(lof_recall_real[1]+0.02,  lof_precision_real[1]+0.035),
                arrowprops=dict(facecolor="purple", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # plt.title("Numeric: Test on real data")
    plt.xlabel('Recall', fontdict={'size' : 30})
    plt.ylabel('Precision', fontdict={'size' : 30})
    plt.xlim(0.0, 1.0)
    plt.ylim(0.4, 1.02)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    plt.grid(linestyle=(0, (5, 10)))

    plt.legend(loc='lower left', prop={'size': 20})
    plt.tight_layout()
    plt.savefig('Numeric: Test on real data.jpg')
    plt.show()



    ################### Test on synthetical data ########################
    plt.figure(figsize=(15, 10), dpi=80)


    # AVH
    plt.plot(k_v4_recall_syn, k_v4_precision_syn, linestyle='solid', linewidth=5, label="AVH", marker='*', markersize=12, c='r')
    plt.annotate("AVH ", 
                xy=(k_v4_recall_syn[9], k_v4_precision_syn[9]), 
                xytext=(k_v4_recall_syn[9]+0.02,  k_v4_precision_syn[9]+0.03),
                arrowprops=dict(facecolor="r", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # azure drift detector(add)
    plt.plot(add_recall_syn[:6], add_precision_syn[:6], linestyle='dashdot', linewidth=5, label="Azure Drift Detector", marker='s', markersize=10, c='darkorange')
    plt.annotate("Azure Drift Detector", 
                xy=(add_recall_syn[4], add_precision_syn[4]), 
                xytext=(add_recall_syn[4]+0.02,  add_precision_syn[4]+0.03),
                arrowprops=dict(facecolor="darkorange", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # # TFDV
    plt.plot(tfdv_recall_syn[1:7], tfdv_precision_syn[1:7], linestyle='dashed', linewidth=5, label="Google TFDV", marker='v', markersize=10, c='dodgerblue')
    plt.annotate("Google TFDV", 
                xy=(tfdv_recall_syn[3], tfdv_precision_syn[3]), 
                xytext=(tfdv_recall_syn[3]-0.15, tfdv_precision_syn[3]-0.04),
                arrowprops=dict(facecolor="dodgerblue", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # Amazon Deequ
    plt.plot(deequ_recall_syn, deequ_precision_syn, linestyle='dotted', linewidth=5, label="Amazon Deequ", marker='o', markersize=10, c='indigo')
    plt.annotate("Amazon Deequ", 
                xy=(deequ_recall_syn[-1], deequ_precision_syn[-1]), 
                xytext=(deequ_recall_syn[-1]+0.03, deequ_precision_syn[-1]-0.04),
                arrowprops=dict(facecolor="indigo", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # # Hypothesis Testing
    plt.plot(hypo_recall_syn[1:20][::3], hypo_precision_syn[1:20][::3], linestyle='dotted', linewidth=5, label="KS-Test", marker='^', markersize=10, c='brown')
    plt.annotate("KS-Test", 
                xy=(hypo_recall_syn[1:20][::3][5], hypo_precision_syn[1:20][::3][5]), 
                xytext=(hypo_recall_syn[1:20][::3][5]+0.02,  hypo_precision_syn[1:20][::3][5]+0.03),
                arrowprops=dict(facecolor="brown", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # XGB
    plt.plot(xgb_recall_syn[100:200:5][12:], xgb_precision_syn[100:200:5][12:], linestyle='dashed', linewidth=5, label="XGBoost", marker='s', markersize=10, c='pink')
    plt.annotate("XGBoost", 
                xy=(xgb_recall_syn[100:200:5][12:][4], xgb_precision_syn[100:200:5][12:][4]), 
                xytext=(xgb_recall_syn[100:200:5][12:][4]+0.02,  xgb_precision_syn[100:200:5][12:][4]+0.035),
                arrowprops=dict(facecolor="pink", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # # Anomaly Detection
    plt.plot(svm_recall_syn, svm_precision_syn, linestyle='dotted', linewidth=3, label="OneClassSVM", marker='v', markersize=10, c='olive')
    plt.annotate("OneClassSVM", 
                xy=(svm_recall_syn[2], svm_precision_syn[2]), 
                xytext=(svm_recall_syn[2]+0.02,  svm_precision_syn[2]+0.035),
                arrowprops=dict(facecolor="olive", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    plt.plot(isof_recall_syn, isof_precision_syn, linestyle='dashdot', linewidth=5, label="Isolaton Forest", marker='*', markersize=10, c='orangered')
    plt.annotate("Isolaton Forest", 
                xy=(isof_recall_syn[5], isof_precision_syn[5]), 
                xytext=(isof_recall_syn[5]-0.17,  isof_precision_syn[5]-0.05),
                arrowprops=dict(facecolor="orangered", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    plt.plot(lof_recall_syn, lof_precision_syn, linestyle='dashed', linewidth=5, label="LocalOutlierFactor", marker='s', markersize=10, c='purple')
    plt.annotate("LocalOutlierFactor", 
                xy=(lof_recall_syn[1], lof_precision_syn[1]), 
                xytext=(lof_recall_syn[1]+0.02,  lof_precision_syn[1]+0.03),
                arrowprops=dict(facecolor="purple", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # plt.title("Numeric: Test on syn data")
    plt.xlabel('Recall', fontdict={'size' : 30})
    plt.ylabel('Precision', fontdict={'size' : 30})
    plt.xlim(0.0, 1)
    plt.ylim(0.4, 1.03)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    plt.grid(linestyle=(0, (5, 10)))

    plt.legend(loc='lower left', prop={'size': 20})
    plt.tight_layout()
    plt.savefig('Numeric: Test on syn data.jpg')
    plt.show()



def plot_category():
    """
    visualize categorical result
    """
    # Google TFDV
    tfdv = TFDV(is_numeric=False)
    tfdv_precision_real, tfdv_precision_syn, tfdv_recall_real, tfdv_recall_syn = tfdv.parse_result_update()

    # Amazon Deequ
    deequ = Deequ()
    deequ_precision_real, deequ_precision_syn, deequ_recall_real, deequ_recall_syn = deequ.parse_result_update(is_numeric=False)

    # Azure Anomaly Detection
    azure_ad = AzureAD(False)
    azure_precision, azure_recall = azure_ad.parse_result_update()

    # AVH-with-stationary
    k_clause = k_v4.KClause(is_numeric=False)
    k_clause.save_base_dir = '../result/k_clause_v4/category_result0227'
    k_v4_precision_real, k_v4_precision_syn, k_v4_recall_real, k_v4_recall_syn = k_clause.parse_result_update()

    # AnomalyDetection
    ad = AnomalyDetection(is_numeric=False, model_name='svm')
    svm_precision_real, svm_precision_syn, svm_recall_real, svm_recall_syn = ad.parse_result_update()

    # Isolation Forest
    ad = AnomalyDetection(is_numeric=False, model_name='isof')
    isof_precision_real, isof_precision_syn, isof_recall_real, isof_recall_syn = ad.parse_result_update()

    # LocalOutlierFactor
    ad = AnomalyDetection(is_numeric=False, model_name='lof')
    lof_precision_real, lof_precision_syn, lof_recall_real, lof_recall_syn = ad.parse_result_update()

    # XGboost
    sm = SupervisedModel(is_numeric=False, model_name='xgb')
    xgb_precision_real, xgb_precision_syn, xgb_recall_real, xgb_recall_syn = sm.parse_result_update()

    # Hypothesis Testing
    hypo = HypoTest(is_numeric=False, verbose=True)
    hypo_precision_real, hypo_precision_syn, hypo_recall_real, hypo_recall_syn = hypo.parse_result_update()


    ################### Test on real data ########################
    plt.figure(figsize=(15, 10), dpi=80)

    # AVH
    plt.plot(k_v4_recall_real, k_v4_precision_real, linestyle='solid', linewidth=5, label="AVH", marker='*', markersize=12, c='r')
    plt.annotate("AVH", 
                xy=(k_v4_recall_real[8], k_v4_precision_real[8]), 
                xytext=(k_v4_recall_real[8]+0.02,  k_v4_precision_real[8]+0.035),
                arrowprops=dict(facecolor="r", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # Azure Anomaly Detector
    plt.plot(azure_recall[1:], azure_precision[1:], linestyle='dashdot', linewidth=5, label="Azure Anomaly Detector", marker='o', markersize=10, c='g')
    plt.annotate("Azure Anomaly Detector", 
                xy=(azure_recall[1], azure_precision[1]), 
                xytext=(azure_recall[1]+0.02, azure_precision[1]+0.03),
                arrowprops=dict(facecolor="g", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # TFDV
    plt.plot(tfdv_recall_real[4:], tfdv_precision_real[4:], linestyle='dashed', linewidth=5, label="Google TFDV", marker='v', markersize=10, c='dodgerblue')
    plt.annotate("Google TFDV", 
                xy=(tfdv_recall_real[4:][6], tfdv_precision_real[4:][6]), 
                xytext=(tfdv_recall_real[4:][6]-0.16, tfdv_precision_real[4:][6]+0.035),
                arrowprops=dict(facecolor="dodgerblue", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # Amazon Deequ
    plt.plot(deequ_recall_real, deequ_precision_real, linestyle='dotted', linewidth=5, label="Amazon Deequ", marker='o', markersize=10, c='indigo')
    plt.annotate("Amazon Deequ", 
                xy=(deequ_recall_real[0], deequ_precision_real[0]), 
                xytext=(deequ_recall_real[0]-0.16, deequ_precision_real[0]-0.05),
                arrowprops=dict(facecolor="indigo", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # Hypothesis Testing
    plt.plot(hypo_recall_real[1:20][::3], hypo_precision_real[1:20][::3], linestyle='dotted', linewidth=5, label="Chi-Squared", marker='^', markersize=10, c='brown')
    plt.annotate("Chi-Squared ", 
                xy=(hypo_recall_real[1:20][::3][-1], hypo_precision_real[1:20][::3][-1]), 
                xytext=(hypo_recall_real[1:20][::3][-1]-0.17,  hypo_precision_real[1:20][::3][-1]-0.04),
                arrowprops=dict(facecolor="brown", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # XGB
    plt.plot(xgb_recall_real[100:200:5][11:19], xgb_precision_real[100:200:5][11:19], linestyle='dashed', linewidth=5, label="XGBoost", marker='s', markersize=10, c='pink')
    plt.annotate("XGBoost", 
                xy=(xgb_recall_real[100:200:5][11:19][0], xgb_precision_real[100:200:5][11:19][0]), 
                xytext=(xgb_recall_real[100:200:5][11:19][0]+0.03,  xgb_precision_real[100:200:5][11:19][0]-0.05),
                arrowprops=dict(facecolor="pink", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # OneClassSVM
    plt.plot(svm_recall_real, svm_precision_real, linestyle='dotted', linewidth=3, label="OneClassSVM", marker='v', markersize=10, c='olive')
    plt.annotate("OneClassSVM", 
                xy=(svm_recall_real[1], svm_precision_real[1]), 
                xytext=(svm_recall_real[1]+0.02,  svm_precision_real[1]+0.035),
                arrowprops=dict(facecolor="olive", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # Isolaton Forest
    plt.plot(isof_recall_real, isof_precision_real, linestyle='dashdot', linewidth=5, label="Isolaton Forest", marker='*', markersize=10, c='orangered')
    plt.annotate("Isolaton Forest", 
                xy=(isof_recall_real[1], isof_precision_real[1]), 
                xytext=(isof_recall_real[1]-0.19,  isof_precision_real[1]-0.04),
                arrowprops=dict(facecolor="orangered", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # LocalOutlierFactor
    plt.plot(lof_recall_real, lof_precision_real, linestyle='dashed', linewidth=5, label="LocalOutlierFactor", marker='s', markersize=10, c='purple')
    plt.annotate("LocalOutlierFactor", 
                xy=(lof_recall_real[1], lof_precision_real[1]), 
                xytext=(lof_recall_real[1]+0.02,  lof_precision_real[1]+0.035),
                arrowprops=dict(facecolor="purple", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # plt.title("Category: Test on real data")
    plt.xlabel('Recall', fontdict={'size' : 30})
    plt.ylabel('Precision', fontdict={'size' : 30})
    plt.xlim(0.0, 1)
    plt.ylim(0.4, 1.03)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    plt.grid(linestyle=(0, (5, 10)))

    plt.legend(loc='lower left', prop={'size': 20})
    plt.tight_layout()
    plt.savefig('Category: Test on real data.jpg')
    plt.show()


    ################### Test on synthetical data ########################
    plt.figure(figsize=(15, 10), dpi=80)

    # AVH-with-stationary
    plt.plot(k_v4_recall_syn, k_v4_precision_syn, linestyle='solid', linewidth=5, label="AVH", marker='*', markersize=12, c='r')
    plt.annotate("AVH", 
                xy=(k_v4_recall_syn[6], k_v4_precision_syn[6]), 
                xytext=(k_v4_recall_syn[6]+0.02,  k_v4_precision_syn[6]+0.035),
                arrowprops=dict(facecolor="r", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # TFDV
    plt.plot(tfdv_recall_syn[4:9], tfdv_precision_syn[4:9], linestyle='dashed', linewidth=5, label="Google TFDV", marker='v', markersize=10, c='dodgerblue')
    plt.annotate("Google TFDV", 
                xy=(tfdv_recall_syn[6], tfdv_precision_syn[6]), 
                xytext=(tfdv_recall_syn[6]+0.04, tfdv_precision_syn[6]+0.045),
                arrowprops=dict(facecolor="dodgerblue", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # Amazon Deequ
    plt.plot(deequ_recall_syn, deequ_precision_syn, linestyle='dotted', linewidth=5, label="Amazon Deequ", marker='o', markersize=10, c='indigo')
    plt.annotate("Amazon Deequ", 
                xy=(deequ_recall_syn[1], deequ_precision_syn[1]), 
                xytext=(deequ_recall_syn[1]-0.17, deequ_precision_syn[1]-0.05),
                arrowprops=dict(facecolor="indigo", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # Hypothesis testing
    plt.plot(hypo_recall_syn[1:20][::3], hypo_precision_syn[1:20][::3], linestyle='dotted', linewidth=5, label="Chi-Squared", marker='^', markersize=10, c='brown')
    plt.annotate("Chi-Squared ", 
                xy=(hypo_recall_syn[1:20][::3][-1], hypo_precision_syn[1:20][::3][-1]), 
                xytext=(hypo_recall_syn[1:20][::3][-1]-0.15,  hypo_precision_syn[1:20][::3][-1]-0.05),
                arrowprops=dict(facecolor="brown", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # XGB
    plt.plot(xgb_recall_syn[100:200:5][12:], xgb_precision_syn[100:200:5][12:], linestyle='dashed', linewidth=5, label="XGBoost", marker='s', markersize=10, c='pink')
    plt.annotate("XGBoost", 
                xy=(xgb_recall_syn[100:200:5][12:][5], xgb_precision_syn[100:200:5][12:][5]), 
                xytext=(xgb_recall_syn[100:200:5][12:][5]-0.10,  xgb_precision_syn[100:200:5][12:][5]-0.05),
                arrowprops=dict(facecolor="pink", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # Anomaly Detection
    plt.plot(svm_recall_syn, svm_precision_syn, linestyle='dotted', linewidth=3, label="OneClassSVM", marker='v', markersize=10, c='olive')
    plt.annotate("OneClassSVM", 
                xy=(svm_recall_syn[1], svm_precision_syn[1]), 
                xytext=(svm_recall_syn[1]+0.02,  svm_precision_syn[1]+0.035),
                arrowprops=dict(facecolor="olive", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # Isolation Forest
    plt.plot(isof_recall_syn[1:], isof_precision_syn[1:], linestyle='dashdot', linewidth=5, label="Isolaton Forest", marker='*', markersize=10, c='orangered')
    plt.annotate("Isolaton Forest", 
                xy=(isof_recall_syn[1:][1], isof_precision_syn[1:][1]), 
                xytext=(isof_recall_syn[1:][1]-0.17,  isof_precision_syn[1:][1]-0.05),
                arrowprops=dict(facecolor="orangered", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)

    # LocalOutlierFactor
    plt.plot(lof_recall_syn, lof_precision_syn, linestyle='dashed', linewidth=5, label="LocalOutlierFactor", marker='s', markersize=10, c='purple')
    plt.annotate("LocalOutlierFactor", 
                xy=(lof_recall_syn[1], lof_precision_syn[1]), 
                xytext=(lof_recall_syn[1]+0.02,  lof_precision_syn[1]+0.035),
                arrowprops=dict(facecolor="purple", shrink=0.01, headlength=10, headwidth=7, width=3),
                fontsize=17)


    # plt.title("Category: Test on syn data")
    plt.xlabel('Recall', fontdict={'size' : 30})
    plt.ylabel('Precision', fontdict={'size' : 30})
    plt.xlim(0, 1)
    plt.ylim(0.4, 1.03)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    plt.grid(linestyle=(0, (5, 10)))


    plt.legend(loc='lower left', prop={'size': 20})
    plt.tight_layout()
    plt.savefig('Category: Test on syn data.jpg')
    plt.show()



if __name__ == '__main__':

    plot_numeric()
    plot_category()
