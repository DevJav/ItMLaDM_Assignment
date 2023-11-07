
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from tabulate import tabulate
import numpy as np, scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import metrics
from toolbox_02450 import mcnemar

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

# Logistic Regression parameters
lambda_interval = np.logspace(-5, 3, 50)
max_iter_param = 1000

# Tree classifier parameters
min_samples = range(2, 100)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(K, shuffle=True)

resume = []
y_predicted_baseline = np.zeros(y.shape)
y_predicted_logistic = np.zeros(y.shape)
y_predicted_tree = np.zeros(y.shape)

for i, (train_index_out, test_index_out) in enumerate(CV.split(X,y)):
    print(f'Outer fold iteration: {i+1}/{K}')

    X_train_out = X[train_index_out]
    y_train_out = y[train_index_out]
    X_test_out = X[test_index_out]
    y_test_out = y[test_index_out]

    for j, (train_index, test_index) in enumerate(CV.split(X_train_out,y_train_out)):
        print(f'    Inner fold iteration: {j+1}/{K}')

        X_train = X_train_out[train_index]
        y_train = y_train_out[train_index]
        X_test = X_train_out[test_index]
        y_test = y_train_out[test_index]

        #!################################
        #!##### Logistic Regression ######
        #!################################
        best_lambda = 0
        best_performance = float('inf')
        for k in range(len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k], max_iter=max_iter_param)
            
            mdl.fit(X_train, y_train)

            y_test_est = mdl.predict(X_test).T
            
            test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

            if test_error_rate < best_performance:
                best_performance = test_error_rate
                best_lambda = lambda_interval[k]

        #!##############################
        #!## Classification Tree #######
        #!##############################
        min_test_error = float('inf')
        best_min_samples = 0
        for k in range(len(min_samples)):
            criterion='gini'
            dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples[k])
            dtc = dtc.fit(X_train,y_train)
            y_test_est = dtc.predict(X_test).T
            test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

            if test_error_rate < min_test_error:
                min_test_error = test_error_rate
                best_min_samples = min_samples[k]

    #!##############################
    #!##### Baseline Model #########
    #!##############################
    # find the most common class in current fold
    y_mean = np.mean(y_train_out)
    if y_mean < 0.5:
        most_common_class = 0
    else:
        most_common_class = 1
    y_predicted = np.ones(y_test_out.shape)*most_common_class
    y_predicted_baseline[test_index_out] = y_predicted

    f, axes = plt.subplots(1, 3, figsize=(20, 5), sharey='row')
    confusion_matrix = metrics.confusion_matrix(y_test_out, y_predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Benign", "Malign"])
    cm_display.plot(cmap=plt.cm.Greens, ax=axes[0])
    plt.title(f'[Baseline] Confusion matrix for outer fold {i+1}')

    E_baseline = np.sum(y_predicted != y_test_out) / len(y_test_out)

    #!##############################
    #! TRAIN LOGISTIC REGRESSION ON OUTER FOLD
    #!##############################
    mdl = LogisticRegression(penalty='l2', C=1/best_lambda, max_iter=max_iter_param)
    mdl.fit(X_train_out, y_train_out)
    y_test_est = mdl.predict(X_test_out).T
    y_predicted_logistic[test_index_out] = y_test_est

    E_logistic = np.sum(y_test_est != y_test_out) / len(y_test_out)

    confusion_matrix = metrics.confusion_matrix(y_test_out, y_test_est)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Benign", "Malign"])
    cm_display.plot(cmap=plt.cm.Blues, ax=axes[1])
    plt.title(f'[Logistic] Confusion matrix for outer fold {i+1}')

    #!##############################
    #! TRAIN CLASSIFICATION TREE ON OUTER FOLD
    #!##############################
    dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=best_min_samples)
    dtc = dtc.fit(X_train_out,y_train_out)
    y_test_est = dtc.predict(X_test_out).T
    y_predicted_tree[test_index_out] = y_test_est

    E_tree = np.sum(y_test_est != y_test_out) / len(y_test_out)

    plt.subplot(1,3,3)
    confusion_matrix = metrics.confusion_matrix(y_test_out, y_test_est)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Benign", "Malign"])
    cm_display.plot(cmap=plt.cm.Reds, ax=axes[2])
    plt.title(f'[Tree] Confusion matrix for outer fold {i+1}')
    plt.show()

    resume.append([i, best_lambda, E_logistic, best_min_samples, E_tree, E_baseline])


print(tabulate(resume, headers=['Fold', 'Best lambda', 'Logistic error', 'Best min_samples', 'Tree error', 'Baseline error']))

#!##############################
#!##### McNemar's Test #########
#!##############################

alpha = 0.05
[thetahat, CI, p] = mcnemar(y, y_predicted_baseline, y_predicted_logistic, alpha=alpha)
print(f'Logistic vs Baseline: theta = {thetahat:.10f}, CI = ({CI[0]:.5f}, {CI[1]:.5f}), p = {p:.10f}')

[thetahat, CI, p] = mcnemar(y, y_predicted_baseline, y_predicted_tree, alpha=alpha)
print(f'Tree vs Baseline: theta = {thetahat:.10f}, CI = ({CI[0]:.5f}, {CI[1]:.5f}), p = {p:.10f}')

[thetahat, CI, p] = mcnemar(y, y_predicted_logistic, y_predicted_tree, alpha=alpha)
print(f'Tree vs Logistic: theta = {thetahat:.10f}, CI = ({CI[0]:.5f}, {CI[1]:.5f}), p = {p:.10f}')