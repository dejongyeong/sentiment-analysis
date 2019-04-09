# import module
import numpy as np
import pandas as pd
from functools import reduce
from matplotlib import pyplot as plt

# import data
names = ['Accuracy', 'Precision', 'Recall', 'F1']
lexicon = pd.read_csv('../results/lexicon_performance_result.csv', delimiter=',', header=None, names=names, skiprows=1)
ml = pd.read_csv('../results/ml_performance_result.csv', delimiter=',', header=None, names=names, skiprows=1)
hybrid = pd.read_csv('../results/hybrid_performance_result.csv', delimiter=',', header=None, names=names, skiprows=1)

# compile list of dataframes
df_list = [lexicon, ml, hybrid]

# merge three dataframes
# reference: https://stackoverflow.com/questions/44327999/python-pandas-merge-multiple-dataframes/44338256
df = reduce(lambda left, right: pd.merge(left, right, how='outer'), df_list)

# transport dataframe
df = df.transpose()

# rename column
df.rename(columns={0: 'Lexicon-Based', 1: 'Machine Learning', 2: 'Hybrid'}, inplace=True)

# change to percentage
columns = ['Lexicon-Based', 'Machine Learning', 'Hybrid']
for index, row in df.iterrows():
    for elem in range(len(columns)):
        df.at[index, columns[elem]] = np.round(row[columns[elem]] * 100, 1)

# visualization
# reference: https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots
fig = plt.figure()
ax = df.plot.bar()
plt.title('Model Evaluation Metrics Comparison of Sentiment Analysis Approaches\n')
plt.ylabel('Score (%)')
plt.xlabel('Model Evaluation Metrics')
plt.xticks(rotation=-360)  # rotate x labels
plt.ylim([0, 100])
for item in ax.patches:  # show value on plot
    ax.annotate(np.round(item.get_height(), decimals=2), (item.get_x() + item.get_width() / 2., item.get_height()),
                fontsize=6,
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig('../results/metrics_comparison.png', format='png', transparent=False)  # save result
plt.show()

# end visualization
