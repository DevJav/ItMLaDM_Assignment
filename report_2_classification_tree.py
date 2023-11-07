# exercise 8.1.2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot)
from toolbox_02450 import rlr_validate
from scipy import stats
from sklearn import tree
from sklearn.model_selection import train_test_split

font_size = 15
plt.rcParams.update({'font.size': font_size})

# Load Matlab data file and extract variables of interest
filename = "data_mean_10.csv"
df = pd.read_csv(filename)
raw_data = df.values
X = raw_data[:, 2:12]
y = raw_data[:, [1]]
attributeNames = list(np.asarray(df.columns[2:12]))
classLabels = raw_data[:, [0]]
# Convert string labels to numeric labels
classLabels = raw_data[:, 1]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape
C = len(classNames)

# standardize data
X = stats.zscore(X)

# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.9, stratify=y)
# Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
# effect of regularization? How does differetn runs of  test_size=.99 compare 
# to eachother?

# # Standardize the training and set set based on training set mean and std
# mu = np.mean(X_train, 0)
# sigma = np.std(X_train, 0)

# X_train = (X_train - mu) / sigma
# X_test = (X_test - mu) / sigma

# Fit regularized logistic regression model to training data to predict 
# the type of wine
min_samples = range(2, 100)
train_error_rate = np.zeros(len(min_samples))
test_error_rate = np.zeros(len(min_samples))
coefficient_norm = np.zeros(len(min_samples))

best_dtc = None
min_test_error = float('inf')

for k in range(len(min_samples)):
    criterion='gini'
    dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples[k])
    dtc = dtc.fit(X_train,y_train)
    y_train_est = dtc.predict(X_train).T
    y_test_est = dtc.predict(X_test).T
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    if test_error_rate[k] < min_test_error:
        min_test_error = test_error_rate[k]
        best_dtc = dtc

fig = plt.figure() 
_ = tree.plot_tree(best_dtc, filled=False,feature_names=attributeNames)
fig.text(0.5,0.9,f'Decision tree for min_samples_split={best_dtc.min_samples_split}', ha='center', fontsize=font_size)

plt.axis('off')
plt.box('off')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
min_test_error = np.min(test_error_rate)
opt_min_samples_idx = np.argmin(test_error_rate)
opt_min_samples = min_samples[opt_min_samples_idx]
ax.plot(opt_min_samples, min_test_error*100, 'o', color='black')
ax.plot(min_samples, train_error_rate*100, '.-')
ax.plot(min_samples, test_error_rate*100,'.-')
ax.set_xlabel('Min samples')
ax.set_ylabel('Error rate (%)')
ax.legend(['Best', 'Train error','Test error'])
ax.grid()
plt.show()
