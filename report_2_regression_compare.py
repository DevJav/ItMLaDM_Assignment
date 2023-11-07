
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot)
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
import torch
from scipy import stats
from tabulate import tabulate
import numpy as np, scipy.stats as st

# Load Matlab data file and extract variables of interest
filename = "data_mean_10.csv"
df = pd.read_csv(filename)
raw_data = df.values
cols = np.r_[1:9, 10:12]
X = raw_data[:, cols]
y = raw_data[:, [9]]
attributeNames = np.asarray(df.columns[cols])

N, M = X.shape

# standardize data
X = stats.zscore(X)
# add offset attribute for w_0
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
attributeNames = [u'offset']+list(attributeNames)
M = M+1

# Linear Regression parameters
lambdas = np.logspace(0, 2, 50)

# ANN parameters
possible_hs = [1, 3, 5, 10]
n_replicates = 3
max_iter = 2000

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(K, shuffle=True)

resume = []
zLinearRegression = []
zANN = []
zBaseline = []

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

        #!##############################
        #!##### Linear Regression ######
        #!##############################
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train.squeeze(), lambdas, K)
        best_lambda = opt_lambda

        #!##############################
        #!########### ANN ##############
        #!##############################
        # We find the best h
        best_h = 0
        best_performance = 1000000

        for n_hidden_units in possible_hs:
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M-1, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

            X_train_ann = torch.Tensor(X_train[:, 1:])
            y_train_ann = torch.Tensor(y_train)
            X_test_ann = torch.Tensor(X_test[:, 1:])
            y_test_ann = torch.Tensor(y_test)

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                            loss_fn,
                                                            X=X_train_ann,
                                                            y=y_train_ann,
                                                            n_replicates=n_replicates,
                                                            max_iter=max_iter)

            y_test_est = net(X_test_ann)

            # Determine errors and errors
            se = (y_test_est.float()-y_test_ann.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test_ann)).data.numpy() #mean

            if mse < best_performance:
                best_performance = mse
                best_h = n_hidden_units

    #!##############################
    #!##### Baseline Model #########
    #!##############################
    y_mean = np.mean(y_train_out)

    #! Squared error loss per observation of baseline model
    E_baseline = np.square(y_test_out - y_mean).sum()/y_test_out.shape[0]
    zBaseline.append(np.square(y_test_out - y_mean))

    #!##############################
    #! TRAIN LINEAR REGRESSION ON OUTER FOLD
    #!##############################
    Xty = X_train_out.T @ y_train_out
    XtX = X_train_out.T @ X_train_out
    lambdaI = best_lambda * np.eye(M)
    lambdaI[0,0] = 0 # remove bias regularization
    w = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    predicted_y = X_test_out @ w

    plt.figure(figsize=(12,8))
    plt.plot(y_test_out, 'o-', label='True value')
    plt.plot(predicted_y, 'x-', label='Predicted value')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Test data: True and predicted value')
    plt.legend()
    plt.show()

    #! Squared error loss per observation of linear regression   
    se_l = np.square(y_test_out.squeeze() - predicted_y)
    E_linear_regression = se_l.sum()/y_test_out.shape[0]
    zLinearRegression.append(se_l)

    #!##############################
    #! TRAIN ANN ON OUTER FOLD
    #!##############################
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M-1, best_h), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(best_h, 1), # n_hidden_units to 1 output neuron
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    X_train_ann = torch.Tensor(X_train_out[:, 1:])
    y_train_ann = torch.Tensor(y_train_out)
    X_test_ann = torch.Tensor(X_test_out[:, 1:])
    y_test_ann = torch.Tensor(y_test_out)

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                    loss_fn,
                                                    X=X_train_ann,
                                                    y=y_train_ann,
                                                    n_replicates=n_replicates,
                                                    max_iter=max_iter)
    
    y_test_est = net(X_test_ann)

    plt.figure(figsize=(10,6))
    plt.plot(y_test_ann, 'ro-', label='True value')
    plt.plot(y_test_est.float().data.numpy(), 'gx-', label='Predicted value')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('ANN: True and predicted value')
    plt.legend()
    plt.show()

    # Determine errors and errors
    se = (y_test_est.float()-y_test_ann.float())**2 # squared error
    E_ann = ((sum(se).type(torch.float)/len(y_test_ann)).data.numpy()[0]) #mean
    zANN.append(se.data.numpy())

    resume.append([i, best_h, E_ann, best_lambda, E_linear_regression, E_baseline])


print(tabulate(resume, headers=['Fold', 'Best h', 'ANN error', 'Best lambda', 'Linear regression error', 'Baseline error']))


#!##############################
#! PAIRED T-TEST ################
#!##############################

alpha = 0.05
#! Linear regression vs ANN
z = []
for i in range(K):
    z.extend(zLinearRegression[i] - zANN[i].squeeze())

CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print(f'Linear regression vs ANN: p={p:.20f}, CI=({CI[0]:.7f},{CI[1]:.7f})')

#! Linear regression vs Baseline
z = []
for i in range(K):
    z.extend(zLinearRegression[i] - zBaseline[i].squeeze())
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print(f'Linear regression vs Baseline: p={p:.20f}, CI=({CI[0]:.7f},{CI[1]:.7f})')

#! ANN vs Baseline
z = []
for i in range(K):
    z.extend(zANN[i].squeeze() - zBaseline[i].squeeze())
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print(f'ANN vs Baseline: p={p:.20f}, CI=({CI[0]:.7f},{CI[1]:.7f})')
