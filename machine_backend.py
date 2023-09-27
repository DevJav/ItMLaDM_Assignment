import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import pandas as pd
import seaborn as sns

##### CONFIGURATION #####

filename = "data.csv"
# Diagnosis is column 1
class_column = 1 # column of the Y
# From 2 to 12 mean
# From 13 to 22 standard error
# From 23 to 32 worst
attribute_start = 2 # column of the first attribute
attribute_end = 12 # column of the last attribute
n_pca_to_use = 4 # number of PCAs to plot
observation_limit = 0 # 0 = no limit
plot_correlation = False # plot correlation between attributes or not (may be really big)

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
corr_matrix = np.corrcoef(X.T)
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

# Create a heatmap with values displayed on each square
sns.heatmap(corr_matrix, cmap='seismic', annot=True, fmt=".2f", cbar=True, square=True,
            xticklabels=attributeNames, yticklabels=attributeNames)

plt.title('Correlation matrix of attributes')
plt.xticks(rotation=90)
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

colors = [ '#D64040',"#40D640"]
plt.figure()
df = pd.DataFrame(X, columns=attributeNames)
df['Diagnostic'] = classLabels
df_long = pd.melt(df, id_vars=['Diagnostic'], var_name='attributes', value_name='values')
custom_palette = sns.color_palette(colors)
sns.violinplot(x="attributes", y="values", hue="Diagnostic", data=df_long, split=True, inner="quart", palette=custom_palette)
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
custom_colors = ["#D64040", "#40D640", "#4040D6", "#D69040", "#D6D640", "#D640A0", "#A040D6"]

colors = ["#40D640", '#D64040']
for c in range(C):
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=0.5, color=colors[c], markersize=7, markeredgecolor='k', markeredgewidth=0.8)
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
plt.grid(zorder=0)
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw, zorder=3)
plt.xticks(r+bw, attributeNames, rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.title(' PCA Component Coefficients')
plt.tight_layout()
plt.show()
