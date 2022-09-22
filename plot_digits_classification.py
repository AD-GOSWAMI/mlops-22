"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# model hyperparams



#1. settin the ranges of hyperparameters


#2. train for every combination of hyper parameter values

#3. train the mode1
#4. compute the accuracy on validations set
#5. Identify the best combination of hyper parameters for which validation set acuracy is the highest
#6. Report the test set accuracyu with that best model


gamma_list = [0.01, 0.003, 0.001, 0.0003, 0.0001]
c_list = [0.1, 0.2, 0.7, 1, 2, 7, 10] 

h_param_list = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

train_frac=0.8
test_frac=0.1
dev_frac=0.1



digits = datasets.load_digits()
n_samples = len(digits.images)
# Classification


data = digits.images.reshape((n_samples, -1))

#image_rescaled = rescale(image, 0.25, anti_aliasing=False)


#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)




#PART: setting up hyperparameter
#hyper_params = {'gamma':GAMMA, 'c':C}
best_acc=-1
best_model=None
best_hyperparams=None
table1=[['Hyper_params', "Train Accuracy %",'Dev Accuracy %', 'Test Accuracy %']]
#table1=[]
for hyper_params in h_param_list:
    #print(hyper_params)   

    #PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    clf.set_params(**hyper_params)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_dev = clf.predict(X_dev)

    current_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)

    if current_acc> best_acc:
        best_acc=current_acc
        best_model=clf
        best_hyperparams=hyper_params            
        #print("Found new best acc :"+str(hyper_params))
        #print("New best val accuracy is:" + str(current_acc))


    predicted_test = clf.predict(X_test)
    test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

    predicted_train = clf.predict(X_train)
    train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)

    predicted_dev = clf.predict(X_dev)
    dev_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)

    table1.append([hyper_params, round(100*train_acc,2),round(100*dev_acc,2), round(100*test_acc,2)])


#print (table1)    
for i in table1:
	print(i)
#print(best_acc)
print("best_hyperparams are: ", best_hyperparams)


predicted_test = best_model.predict(X_test)
test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

predicted_train = best_model.predict(X_train)
train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)

predicted_dev = best_model.predict(X_dev)
dev_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)

print("train acc: " + str(round(100*train_acc,2))+" %")
print("dev acc: " + str(round(100*dev_acc,2))+" %")
print("test acc: " + str(round(100*test_acc,2))+" %")

