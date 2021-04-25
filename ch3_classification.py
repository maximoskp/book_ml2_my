# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 07:35:58 2021

@author: user
"""

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=False)

print( repr(mnist.keys()) )

# %% 

X , y = mnist['data'] , mnist['target']

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

some_digit = X.iloc[0].to_numpy().astype(np.uint8)
# some_digit = X.iloc[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow( some_digit_image , cmap='binary' )
plt.axis( 'off' )
plt.show()

y = y.astype( np.uint8 )

# %% train -  test split

X_train , X_test , y_train , y_test = X.iloc[:60000] , X.iloc[60000:] , y[:60000] , y[60000:]

# %% identify digit 5

y_train_5 = (y_train == 5) # binary
y_test_5 = (y_test == 5)

# %%

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier( random_state=42 )
sgd_clf.fit( X_train , y_train_5 )

# %% test classifier

i = 0

some_digit = X.iloc[i].to_numpy().astype(np.uint8)
# some_digit = X.iloc[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow( some_digit_image , cmap='binary' )
plt.axis( 'off' )
plt.title( str( sgd_clf.predict( [some_digit] ) ) )
plt.show()

# %% cross validation

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold( n_splits=3, random_state=42 , shuffle=True )

for train_index , test_index in skfolds.split( X_train , y_train_5 ):
    clone_clf = clone( sgd_clf )
    X_train_folds = X_train.iloc[ train_index ]
    y_train_folds = y_train_5[ train_index ]
    X_test_folds = X_train.iloc[ test_index ]
    y_test_folds = y_train_5[ test_index ]
    
    clone_clf.fit( X_train_folds , y_train_folds )
    y_pred = clone_clf.predict( X_test_folds )
    n_correct = sum( y_pred == y_test_folds )
    print( n_correct / len(y_pred) )

# %% cross validation

from sklearn.model_selection import cross_val_score

cvs = cross_val_score( sgd_clf , X_train, y_train_5 , cv=3 , scoring='accuracy' )

# %% always guessing that 5 is not the answer gives 90% accuracy

from sklearn.base import BaseEstimator

class Never5Classifier( BaseEstimator ):
    def fit( self , X , y=None):
        pass
    def predict( self, X ):
        return np.zeros( (len(X) , 1 ) , dtype=bool )

never_5_clf = Never5Classifier()
cvsn5 = cross_val_score( never_5_clf , X_train , y_train_5 , cv=3 , scoring='accuracy' )

# %% better look at the confusion matrix

from sklearn.model_selection import cross_val_predict

# runs all data throughout the folds - finally has a prediction for each data point
y_train_pred = cross_val_predict( sgd_clf , X_train , y_train_5 , cv=3 )

from sklearn.metrics import confusion_matrix
cm = confusion_matrix( y_train_5 , y_train_pred  )

# %% assume perfect performance, the confusion matrix should look like this:
y_train_perfect_predictions = y_train_5
cm_perfect = confusion_matrix( y_train_5 , y_train_perfect_predictions )

# %% in combination with confusion matrix - precision and recall 

from sklearn.metrics import precision_score , recall_score, f1_score

pr = precision_score( y_train_5 , y_train_pred )
rc = recall_score( y_train_5 , y_train_pred )

f1 = f1_score( y_train_5 , y_train_pred )

# %% classifier decision threashold as a function of precision and recall  values

y_scores = sgd_clf.decision_function( [some_digit] )
# classifier decides based on some threshold
threshold = 0
y_some_digit_pred = ( y_scores > threshold )
# if threshold was, e.g., 8000, it would have missed it

# %% decide on the threshold based on cross_val_predict()

y_scores = cross_val_predict( sgd_clf , X_train , y_train_5 , cv=3 , method='decision_function' )

from sklearn.metrics import precision_recall_curve

precisions , recalls , thresholds = precision_recall_curve( y_train_5 , y_scores )

# %% plot curve

def plot_precision_recall_vs_threshold( p , r , t ):
    plt.plot( t , p[:-1]  , 'b--' , label='Precision' )
    plt.plot( t , r[:-1]  , 'g-' , label='Recall' )
    plt.legend()

plot_precision_recall_vs_threshold( precisions, recalls , thresholds )
plt.show()

# %% decide with a target precision in mind

plt.plot( recalls , precisions )
plt.show()

# aim for 90% precision
threshold_90_precision = thresholds[ np.argmax( precisions >= 0.9 ) ]
y_train_pred_90 = ( y_scores >= threshold_90_precision )

pr = precision_score( y_train_5 , y_train_pred_90 )
rc = recall_score( y_train_5 , y_train_pred_90 )

# %% ROC curve: true positive rate (recall) against false positive rate

from sklearn.metrics import roc_curve

fpr , tpr , thresholds = roc_curve( y_train_5 , y_scores )

def plot_roc_curve( fpr , tpr , label='ROC' ):
    plt.plot( fpr , tpr , linewidth=2 , label=label )
    plt.plot( [0,1] , [0,1] , 'k--' )
    plt.legend()

plot_roc_curve( fpr , tpr )
plt.show()
# better as far away as possible from the diagonal

# for classifier comparison, measure the area under the curve AUC
# perfect classifier: 1, random classifier: 0

from sklearn.metrics import roc_auc_score

roc_sgd = roc_auc_score( y_train_5 , y_scores )
print('roc_sgd: ' + str(roc_sgd) )

# %% let's compare with the random forest classifier

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier( random_state=42 )
y_probas_forest = cross_val_predict( forest_clf , X_train , y_train_5, cv=3, method='predict_proba' )

# %%
# probability of positive class
y_score_forest = y_probas_forest[:,1]

fpr_forest , tpr_forest , thresholds_forest = roc_curve( y_train_5 , y_score_forest )

plt.plot( fpr , tpr , 'b:' , label='SGD' )
plot_roc_curve( fpr_forest , tpr_forest , 'Random Forest' )
plt.show()

roc_forest = roc_auc_score( y_train_5 , y_score_forest )
print('roc_sgd: ' + str(roc_forest) )

# %% Multiclass classification

# OvR: one versus rest - N classifiers - one for each class
# OvO: one versus one - N(N-1) classifiers - one for each pair of classes
# sklearn decides depending on the algorithm

from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit( X_train , y_train ) # OvO is selected by default for SVM
tmp_pred = svm_clf.predict( [some_digit] )
print(tmp_pred)

# %% the decision_function returns one value for each class
some_digit_scores = svm_clf.decision_function( [some_digit] )
print( 'scores: ' + repr( some_digit_scores ) )

# and the highest score corresponds to the winning class
# automatic-random ordinal assignment of classes to numbers - happens to match
# class labels in this example
print( 'classes: ' + repr( svm_clf.classes_ ) )
print( 'argmax: ' + str( np.argmax( some_digit_scores ) ) )

# %% OvR can be forced - for SVM it's much slower

from sklearn.multiclass import OneVsRestClassifier

ovr_svm_clf = OneVsRestClassifier( SVC() )
ovr_svm_clf.fit( X_train , y_train )
tmp_pred = ovr_svm_clf.predict( [some_digit] )
print(tmp_pred)

# check how many clasifiers have been created
print('number of classifiers: ' + str( len( ovr_svm_clf.estimators_ ) ) )

# %% SGD or Random Forest can handle multiple classes

sgd_clf.fit( X_train , y_train )
tmp_pred = sgd_clf.predict( [some_digit] )
print(tmp_pred)

# get scores
some_digit_scores = sgd_clf.decision_function( [some_digit] )

# %% check with cross validation

cvs = cross_val_score( sgd_clf , X_train , y_train , cv=3 , scoring='accuracy' )
print( 'cross validation scores: ' + str( repr( cvs ) ) )

# %% scaling input increases performance

from sklearn.preprocessing import StandardScaler
import numpy as np # REMOVE

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform( X_train.astype(np.float64) )
cvs = cross_val_score( sgd_clf , X_train_scaled , y_train , cv=3 , scoring='accuracy' )
print( 'cross validation scores: ' + str( repr( cvs ) ) )

# %% Error analysis

import matplotlib.pyplot as plt # REMOVE
from sklearn.model_selection import cross_val_predict # REMOVE

# get predictions and form confusion matrix
y_train_pred = cross_val_predict( sgd_clf , X_train_scaled , y_train , cv=3 )

from sklearn.metrics import confusion_matrix # REMOVE

conf_mx = confusion_matrix( y_train , y_train_pred )

plt.matshow( conf_mx , cmap=plt.cm.gray_r )
plt.show()

# %% why 5 is darker?

# could be that there are less 5s or that they are classified more poorly
# remove the "number of instances" component

row_sums = conf_mx.sum( axis=1 , keepdims=True )
norm_conf_mx = conf_mx / row_sums

# fill diagonal with zeros to make error effect more vivid
np.fill_diagonal( norm_conf_mx , 0 )
plt.matshow( norm_conf_mx , cmap=plt.cm.gray_r )
plt.show()

# many digits are misclassified as 8s

# %% Multilabel classification

# e.g. number can be >=7 and/or odd

from sklearn.neighbors import KNeighborsClassifier

y_train_large = ( y_train >= 7 )
y_train_odd = ( y_train % 2 == 1 )
y_multilabel = np.c_[ y_train_large , y_train_odd ]

knn_clf = KNeighborsClassifier()
knn_clf.fit( X_train , y_multilabel )

# %% predict

tmp_pred = knn_clf.predict( [some_digit] )
print( repr(tmp_pred) )

y_train_knn_pred = cross_val_predict( knn_clf , X_train , y_multilabel , cv=3 )

tmp_f1 = f1_score( y_multilabel , y_train_knn_pred , average='macro' )
# or average='weighted', for accounting for element count
print('tmp_f1: ' + str(tmp_f1) )

# %% denoising example for multioutput classification

# many values per label - many labels

# train set
noise = np.random.randint( 0, 100 , (len(X_train) , 784) )
X_train_mod = X_train + noise # noisy images as inputs
# test set
noise = np.random.randint( 0, 100 , (len(X_test) , 784) )
X_test_mod = X_test + noise # noisy images as inputs
# output is the original images
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit( X_train_mod , y_train_mod )

# %% check results

def plot_two_digits( d1 , d2 ):
    sd1 = d1.reshape(28,28)
    sd2 = d2.reshape(28,28)
    plt.subplot(121)
    plt.imshow( sd1 , cmap='binary' )
    plt.axis( 'off' )
    plt.subplot(122)
    plt.imshow( sd2 , cmap='binary' )
    plt.axis( 'off' )
    plt.show()

some_index = 1
clean_digit = knn_clf.predict( [X_test_mod.iloc[some_index]] )

plot_two_digits( X_test_mod.iloc[ some_index ].to_numpy() , clean_digit )