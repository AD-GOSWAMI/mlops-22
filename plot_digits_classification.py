"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
from sklearn import datasets, svm, metrics
from sklearn import tree 
import pdb

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
    test_bias,
    test_class,
)



from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb = get_all_h_param_comb(params)


# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits





svm_model_metric=[]
DT_model_metric=[]


x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
data, label, train_frac, dev_frac)   

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
# define the evaluation metric
metric = metrics.accuracy_score
SVM_best_model, SVM_best_metric, SVM_best_h_params = h_param_tuning(
    h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric
)
predicted = SVM_best_model.predict(x_test)
SVM_test_metric = metric(y_pred=predicted, y_true=y_test)

test_bias(SVM_best_model,x_test, y_test)

test_class(SVM_best_model,x_test, y_test)

