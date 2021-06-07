# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 07:51:30 2021

@author: user
"""

# %% linear regression

import numpy as np
import matplotlib.pyplot as plt

X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)

plt.plot( X , y, '.' )
plt.axis([0,2,0,15])
plt.show()

# %% apply normal equation

# include bias
X_b = np.c_[ np.ones( (100, 1) ) , X ]
theta_best = np.linalg.inv( X_b.T.dot( X_b ) ).dot( X_b.T ).dot( y )

print( 'theta_best: ' + repr(theta_best) )
# best would be: theta_best = [4, 3]

# make new prediction
X_new = np.array([ [0] , [2] ])
X_new_b = np.c_[ np.ones((2,1)) , X_new ]
y_predict = X_new_b.dot( theta_best )

print( 'y_predict: ' + repr(y_predict) )

plt.plot( X_new , y_predict , 'r-' , label='Predictions')
plt.plot( X , y, 'b.' , label='Data' )
plt.axis([0,2,0,15])
plt.legend()
plt.show()

# %% with sklearn

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit( X , y )

print( lin_reg.intercept_ )
print( lin_reg.coef_ )

lin_predict = lin_reg.predict( X_new )
print('lin_predict: ' + repr(lin_predict))

# %% 

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b , y , rcond=1e-6)

print('theta_best_svd: ' + repr(theta_best_svd))

# based on pseudoinverse computation
X_b_pseudo = np.linalg.pinv(X_b)
theta_best_pseudo = X_b_pseudo.dot(y)
print('theta_best_pseudo: ' + repr(theta_best_pseudo))

# %% gradient descend - linear regression - batch GD

eta = 0.1 # learning rate
n_iterations = 50
m = 100

theta = np.random.randn(2,1) # random starting parameters

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot( X_b.dot( theta ) - y )
    theta = theta - eta * gradients
    print( 'theta :' + repr(theta) )
    y_predict = X_new_b.dot( theta )
    plt.plot( X_new , y_predict , 'r-' , label='Predictions')
    plt.plot( X , y, 'b.' , label='Data' )
    plt.axis([0,2,0,15])
    plt.pause(0.01)
    plt.show()

# %% stochastic gradient descend with decreasing learning rate

n_epochs= 10
t0, t1 = 5, 50 # for adjusting learning rate

def learning_schedule( t ):
    return t0/(t + t1)

theta = np.random.randn(2,1) # random initialisation

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[ random_index:(random_index+1) ]
        yi = y[ random_index:(random_index+1) ]
        gradients = 2 * xi.T.dot( xi.dot(theta) - yi )
        eta = learning_schedule( epoch*m + i )
        theta = theta - eta*gradients
    print( 'theta :' + repr(theta) )
    y_predict = X_new_b.dot( theta )
    plt.plot( X_new , y_predict , 'r-' , label='Predictions')
    plt.plot( X , y, 'b.' , label='Data' )
    plt.axis([0,2,0,15])
    plt.pause(0.01)
    plt.show()

# %% sklearn SGD linear regressor

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor( max_iter=1000, tol=1e-3, penalty=None, eta0=0.1 )
sgd_reg.fit( X , y.ravel() )

print( sgd_reg.intercept_ )
print( sgd_reg.coef_ )

# %% polynomial regression

m = 100
X = 6*np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot( X , y , '.b' )
plt.axis([-3,3,0,10])
plt.show()

# %% 

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures( degree=2, include_bias=False )
X_poly = poly_features.fit_transform( X )

# X_poly includes one column with X and another with X**2

# fit linear regression to a squared x axis
lin_reg = LinearRegression()
lin_reg.fit( X_poly, y )
print( lin_reg.intercept_ )
print( lin_reg.coef_ )

X_new = np.arange( -3, 3, 0.01 )
y_predict = lin_reg.intercept_ + lin_reg.coef_[0][0]*X_new + lin_reg.coef_[0][1]*X_new**2

plt.plot( X_new , y_predict , 'r-' , label='Predictions')
plt.plot( X , y, 'b.' , label='Data' )
plt.axis([-3,3,0,10])
plt.legend()
plt.show()

# %% Learning curves

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves( model , X , y ):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors , val_errors = [] , []
    for m in range(1, len( X_train )):
        model.fit( X_train[:m], y_train[:m] )
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append( mean_squared_error(y_train[:m], y_train_predict) )
        val_errors.append( mean_squared_error(y_val, y_val_predict) )
    plt.plot( np.sqrt( train_errors ) , 'r-+', linewidth=2, label='train' )
    plt.plot( np.sqrt( val_errors ) , 'b-', linewidth=3, label='val' )
    plt.legend()
    plt.show()

lin_reg = LinearRegression()
plot_learning_curves( lin_reg , X, y )

# test polynomial regression

from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
    ('poly_features', PolynomialFeatures( degree=10, include_bias=False )),
    ('lin_reg', LinearRegression())
])

plot_learning_curves( polynomial_regression , X, y)

# %% Ridge regression: keeping weights of the polynomial model

from sklearn.linear_model import Ridge

ridge_reg = Ridge( alpha=1, solver='cholesky' )
ridge_reg.fit( X , y )

y_ridge_predict = ridge_reg.predict( [[1.5]] )
print('y_ridge_predict: ' + repr(y_ridge_predict))

y_lin_predict = lin_reg.predict( [[1.5]] )
print('y_lin_predict: ' + repr(y_lin_predict))

sgd_reg.fit( X , y.ravel() )
y_sgd_predict = sgd_reg.predict( [[1.5]] )
print('y_sgd_predict: ' + repr(y_sgd_predict))

# sgd with l2 norm - i.e. ridge regression
sgd_reg = SGDRegressor( penalty='l2' )
sgd_reg.fit( X , y.ravel() )
y_sgd_predict = sgd_reg.predict( [[1.5]] )
print('y_sgd_predict - l2: ' + repr(y_sgd_predict))

# %% lasso regression - or l1

from sklearn.linear_model import Lasso

lasso_reg = Lasso( alpha=0.1 )
lasso_reg.fit( X , y )
y_lasso_predict = lasso_reg.predict( [[1.5]] )
print('y_lasso_predict: ' + repr(y_lasso_predict))

# %% elasticnet: combination of ridge and lasso

from sklearn.linear_model import ElasticNet

# l1_ratio: how much of lasso
elastic_net = ElasticNet( alpha=0.1, l1_ratio=0.5)
elastic_net.fit( X , y )
y_elastic_predict = elastic_net.predict( [[1.5]] )
print('y_elastic_predict: ' + repr(y_elastic_predict))

# %% early stopping
# keep the best model when validation error starts increasing

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

# prepare the data
poly_scaler = Pipeline([
    ('poly_features', PolynomialFeatures(degree=90, include_bias=False)),
    ('std_scaler', StandardScaler())
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train_poly_scaled = poly_scaler.fit_transform( X_train )
X_val_poly_scaled = poly_scaler.fit_transform( X_val )

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate='constant', eta0=0.0005)

minimum_val_error = float('inf')
best_epoch = None
best_model = None

for epoch in range(1000):
    sgd_reg.fit( X_train_poly_scaled, y_train.ravel() ) # continues where it left off
    y_val_predict = sgd_reg.predict( X_val_poly_scaled )
    val_error = mean_squared_error( y_val , y_val_predict )
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone( sgd_reg )

# %% logistic regression

# load dataset
from sklearn import datasets

iris = datasets.load_iris()
print( repr( list( iris.keys() ) ) )

X = iris['data'][:, 3:] # pedal width
y = (iris['target'] == 2).astype(np.int32) # 1 if Iris-Virginica

# %% model

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit( X , y )

# fake pedal widths of size from 0 to 3 cm
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba( X_new )
plt.plot( X_new , y_proba[:,1], 'g-', label='Iris-Virginica' )
plt.plot( X_new , y_proba[:,0], 'b--', label='Not Iris-Virginica' )
plt.legend()
plt.show()

# %% decision boundary around 1.6
y_example_predict = log_reg.predict([ [1.7] , [1.5] ])
print('y_example_predict: ' + repr(y_example_predict))
y_example_proba = log_reg.predict_proba([ [1.7] , [1.5] ])
print('y_example_proba: ' + repr(y_example_proba))

# %% two dimentions of input

print( repr( list( iris['feature_names'] ) ) )
X = iris['data'][:, 2:] # pedal length and width

log_reg = LogisticRegression()
log_reg.fit( X , y )

# %% softmax regression

X = iris['data'][:,(2,3)]
y = iris['target']

softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
softmax_reg.fit( X , y )

# %% predictions

y_softmax_prediction = softmax_reg.predict([[5,2]])
print('y_softmax_prediction: ' + repr(y_softmax_prediction))
y_softmax_probas = softmax_reg.predict_proba([[5,2]])
print('y_softmax_probas: ' + repr(y_softmax_probas))