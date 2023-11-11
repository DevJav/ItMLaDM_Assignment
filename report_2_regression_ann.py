
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot)
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
import torch
from scipy import stats

# Load Matlab data file and extract variables of interest
filename = "data_mean_10.csv"
df = pd.read_csv(filename)
raw_data = df.values
cols = np.r_[1:9, 10:12]
X = raw_data[:, cols]
y = raw_data[:, [9]]
attributeNames = np.asarray(df.columns[cols])
print(attributeNames)

N, M = X.shape

# Normalize data
X = stats.zscore(X)

possible_hs = [1, 2, 3]
n_replicates = 1
max_iter = 2000

# K-fold crossvalidation
K = 5
CV = model_selection.KFold(K, shuffle=True)


# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

best_performance_error = []
best_performance_learning_curve = []
best_performance_net = []

for n_hidden_units in possible_hs:
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    print(f'Training model of type:\n\n{str(model())}\n')
    errors = [] # make a list for storing generalizaition error in each loop
    learning_curves = [] # make a list for storing learning curves in each loop
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    

        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train,
                                                        y=y_train,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)

        print(f'\n\tBest loss: {final_loss}\n')

        # Determine estimated class labels for test set
        y_test_est = net(X_test)

        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        errors.append(float(mse)) # store error rate for current CV fold 
        print(errors)

        # Display the learning curve for the best net in the current fold
        learning_curves.append(learning_curve)

    best_of_fold_idx = np.argmin(errors)
    best_performance_error.append(errors[best_of_fold_idx])
    best_performance_learning_curve.append(learning_curves[best_of_fold_idx])
    best_performance_net.append(copy.deepcopy(net))

summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
for i in range(len(possible_hs)):
    h, = summaries_axes[0].plot(best_performance_learning_curve[i], color=color_list[i])
    h.set_label('Hidden units = {0}'.format(possible_hs[i]))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves for different hidden units')
summaries_axes[0].legend()

summaries_axes[1].bar(np.arange(1, len(possible_hs)+1), np.squeeze(np.array(best_performance_error)),  color=color_list)
summaries_axes[1].set_xlabel('Hidden units')
summaries_axes[1].set_xticks(np.arange(1, len(possible_hs)+1), possible_hs)
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')