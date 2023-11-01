import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import pandas as pd
import sklearn.linear_model as lm
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show

filename = "data_mean.csv"

df = pd.read_csv(filename)

cols = range(2, 11)
raw_data = df.values
X = raw_data[:, cols]
X = X.astype(float)
X = stats.zscore(X)

attributeNames = np.asarray(df.columns[cols])

y = raw_data[:, 1]

model = lm.LogisticRegression(max_iter=1000)
model.fit(X, y)

# Classify as Malignant(1) or Benign(0)
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0] 

x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5]).reshape(1,-1)
x_class = model.predict_proba(x)[0,0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nProbability of given sample being malign: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure()
benign_ids = np.nonzero(y=='B')[0].tolist()
plot(benign_ids, y_est_white_prob[benign_ids], 'og', markersize=5, markeredgecolor='k', markeredgewidth=0.8, alpha=.82)
malign_ids = np.nonzero(y=='M')[0].tolist()
plot(malign_ids, y_est_white_prob[malign_ids], 'or', markersize=5, markeredgecolor='k', markeredgewidth=0.8, alpha=.82)
xlabel('Data object'); ylabel('Predicted prob. of class Malign')
legend(['Malign', 'Benign'])
ylim(-0.01,1.5)

show()