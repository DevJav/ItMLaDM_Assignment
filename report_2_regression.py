import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import pandas as pd
import sklearn.linear_model as lm

filename = "data_mean.csv"

df = pd.read_csv(filename)

cols = range(1, 9)
raw_data = df.values
X_prov = raw_data[:, cols]
X = np.array([1 if item == 'M' else 0 for item in X_prov[:,0]])
X = np.vstack((X, X_prov[:,1:].T)).T
X = X.astype(float)
X = stats.zscore(X)

attributeNames = np.asarray(df.columns[cols])

y = raw_data[:, 10]

model = lm.LinearRegression()
model.fit(X, y)

y_est = model.predict(X)
residual = y_est-y

errors = abs(y - y_est)

from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('gr',["g", "y", "r"], N=256) 

# Display scatter plot and histogram
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.scatter(y, y_est, c=errors, cmap=cmap, linewidths=0.5, edgecolors='k')
plt.colorbar(label='Absolute Error')
plt.xlabel('Symmetry (true)'); plt.ylabel('Symmetry (estimated)')

plt.subplot(2,1,2)
plt.hist(residual, 40, color='skyblue', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()