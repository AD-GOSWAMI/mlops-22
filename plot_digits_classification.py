
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from tabulate import tabulate

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

from utils import train_dev_test_split

gamma_list = [0.02, 0.002, 0.0006, 0.0001]
c_list = [0.1, 0.3, 1, 10] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]





digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Classification

# flatten the images
n_samples = len(digits.images)


data = digits.images.reshape((n_samples, -1))

#image_rescaled = rescale(image, 0.25, anti_aliasing=False)

train_frac, test_frac, dev_frac =0.5,0.2,0.3

#random_state_seed=10
random_state_seed=30

#X_train, X_dev_test, y_train, y_dev_test = train_test_split(
#    data, digits.target, test_size=dev_test_frac, shuffle=True,random_state_seed
#)
#X_test, X_dev, y_test, y_dev = train_test_split(
#    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True, random_state_seed
#)
label=digits.target
X_train, y_train, X_dev, y_dev, X_test, y_test=train_dev_test_split(data, label, train_frac, dev_frac, random_state_seed)


#PART: setting up hyperparameter
#hyper_params = {'gamma':GAMMA, 'c':C}
best_acc=-1
best_model=None
best_hyperparams=None

table1=[['Hyper_params', "Train Accuracy %",'Dev Accuracy %', 'Test Accuracy %']]
table2=[]
#min_acc=[0,0,0]
#max_acc=[0,0,0]

for hyper_params in h_param_comb:
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
    table2.append([train_acc,dev_acc,test_acc])

    

#min_acc=[0,0,0]
#max_acc=[0,0,0]
#min_acc=min(np.array(table2),1)

print(tabulate(table1,headers='firstrow',tablefmt='grid'))    
print("Min Accuracy (train-dev-test): ",np.min(np.array(table2),axis=0))
print("Max Accuracy (train-dev-test): ",np.max(np.array(table2),axis=0))
print("Median Accuracy (train-dev-test): ",np.median(np.array(table2),axis=0))
print("Mean Accuracy (train-dev-test): ",np.mean(np.array(table2),axis=0))
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



