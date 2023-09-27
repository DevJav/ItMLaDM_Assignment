import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

##### CONFIGURATION #####

filename = "data.csv"
class_column = 1 # column of the Y
attribute_start = 22 # column of the first attribute
attribute_end = 32 # column of the last attribute
n_pca_to_use = 2 # number of PCAs to plot
observation_limit = 0 # 0 = no limit
plot_correlation = False # plot correlation between attributes

#########################

df = pd.read_csv(filename)

raw_data = df.values
if (observation_limit != 0):
    raw_data = raw_data[range(observation_limit), :]
cols = range(attribute_start, attribute_end)
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])

# Convert string labels to numeric labels
classLabels = raw_data[:, class_column]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape

C = len(classNames)

#! #################################################
#! ############## Plot correlation #################
#! #################################################
if plot_correlation:
    subplot_size = attribute_end - attribute_start
    i = 0
    j = 1
    first_iteration = True
    plt.figure()
    for colx in cols:
        for coly in cols:
            for c in range(C):
                class_mask = y==c
                plt.subplot(subplot_size, subplot_size, j)
                plt.plot(X[class_mask,colx - attribute_start], X[class_mask,coly- attribute_start], 'o', alpha=0.5)
                if coly == cols[-1]:
                    plt.xlabel(attributeNames[colx - attribute_start])
                if first_iteration:
                    plt.ylabel(attributeNames[coly - attribute_start])
            plt.grid()
            j = j + subplot_size
        i = i + 1
        j = 1 + i
        first_iteration = False
    plt.tight_layout()
    plt.show()

    plt.figure()
    for c in range(C):
        class_mask = y==c
        plt.plot(X[class_mask,1], X[class_mask,2], 'o', alpha=0.5)
    plt.tight_layout()
    plt.show()

#! convert X values to float
X = X.astype(float)

#! #################################################
#! # basic summary statistics of the attributes ####
#! #################################################

means = X.mean(axis=0)
stds = X.std(axis=0)
medians = np.median(X, axis=0)
mins = X.min(axis=0)
maxs = X.max(axis=0)

# save the summary statistics in a file csv
summary = np.vstack((means, stds, medians, mins, maxs))
summary = np.transpose(summary)
summary_df = pd.DataFrame(summary, columns=['mean', 'std', 'median', 'min', 'max'], index=attributeNames)
summary_df.to_csv('summary.csv')

#! #################################################
#! # Correlation between attributes #################
#! #################################################
# Compute and plot pairwise correlation coefficients
plt.figure()
plt.imshow(np.corrcoef(X.T), cmap='gray')
plt.colorbar()
plt.xticks(range(M), attributeNames, rotation=90)
plt.yticks(range(M), attributeNames)
plt.title('Correlation matrix of attributes')
plt.tight_layout()
plt.show()

#! Standardize the data
X = stats.zscore(X)

#! PCA by computing SVD of X
U,S,V = np.linalg.svd(X,full_matrices=False)
V = V.T

#! #################################################
#! # Violin plot of the attributes ##################
#! #################################################

# violin plot of attributes with class labels as hue (color)
import seaborn as sns
plt.figure()
df = pd.DataFrame(X, columns=attributeNames)
df['Diagnostic'] = classLabels
df_long = pd.melt(df, id_vars=['Diagnostic'], var_name='attributes', value_name='values')
sns.violinplot(x="attributes", y="values", hue="Diagnostic", data=df_long, split=True, inner="quart")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#! #################################################
#! ########## Plot variance explained ##############
#! #################################################
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()
threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.tight_layout()
plt.show()

# Project the centered data onto principal component space
Z = X @ V

#! #################################################
#! ######## 2D PCA plot of first 2 PCAs ############
#! #################################################
# Indices of the principal components to be plotted
i = 0
j = 1
# Plot PCA of the data
f = plt.figure()
plt.title('data: PCA')
#Z = array(Z)
for c in range(C):
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=0.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.grid()
plt.axis('equal')
# Output result to screen
plt.tight_layout()
plt.show()

#! #################################################
#! ############# PCAs coefficients #################
#! #################################################
# Plot attribute coefficients in first X principal component space in bars
n_pca_to_use = min(n_pca_to_use, V.shape[1])
pcs = range(n_pca_to_use)
legendStrs = [f'PC{str(e + 1)}' for e in pcs]
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames, rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title(' PCA Component Coefficients')
plt.tight_layout()
plt.show()
