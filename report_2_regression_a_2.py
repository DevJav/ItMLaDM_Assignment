# exercise 8.1.2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot)
from toolbox_02450 import rlr_validate
from scipy import stats

font_size = 15
plt.rcParams.update({'font.size': font_size})

# Load Matlab data file and extract variables of interest
filename = "data_mean_10.csv"
df = pd.read_csv(filename)
raw_data = df.values
cols = np.r_[3:11]
X = raw_data[:, cols]
y = raw_data[:, [2]]
attributeNames = list(np.asarray(df.columns[cols]))
N, M = X.shape
print(attributeNames)


X = stats.zscore(X)
# add offset attribute for w_0
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
attributeNames = [u'offset']+attributeNames
M = M+1

K = 10

lambdas = np.power(10.,range(-5,8))
lambdas = np.logspace(-1.5, 5, 50)

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, K)

figure(0, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas, mean_w_vs_lambda.T[:,1:], '.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
legend(attributeNames[1:], loc='best', fontsize="small")
grid()

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas, train_err_vs_lambda.T,'b.-',lambdas, test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
show()

CV = model_selection.KFold(K, shuffle=True)
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # remove bias regularization
    w = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    predicted_y = X_test @ w
    print("Test error: ", np.square(y_test - predicted_y).sum()/y_test.shape[0])

plt.figure(figsize=(12,8))
plt.plot(y_test, 'o-', label='True value')
plt.plot(predicted_y, 'x-', label='Predicted value')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Test data: True and predicted value')
plt.legend()
plt.show()
