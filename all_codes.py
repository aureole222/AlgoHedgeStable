from dtw import dtw
import pandas_datareader.data as web
import numpy as np
from numpy.linalg import norm
from numpy import array
import matplotlib.pyplot as plt
import datetime
from fastdtw import fastdtw
import seaborn as sns
import pandas as pd
import numpy as np
import datetime 
import os
import matplotlib.pyplot as plt


maturity   = 5

''' convert daily data to monthly frequency '''
this_df=pd.read_csv('C:/Users/richard/Documents/stable/algorithm hedging/usable index/daily_index.csv',index_col=0)
this_df.index = pd.DatetimeIndex(this_df.index)
groups = this_df.resample('M')
monthly_data = groups.mean()
#monthly_data = pd.concat([monthly_data,df],axis=1)
''' calculate return ratio per future during maturity period'''
return_ratio=(monthly_data.iloc[maturity:,:]-monthly_data.iloc[:-maturity,:].values).divide(monthly_data.iloc[:-maturity,:].values)


sns.set_style("whitegrid", {'font_scale':0.5,
                           'font.scale':0.5,
                           'xtick.color':'#A5A5A5',
                           'ytick.color':'#A5A5A5',
                           'axes.linewidth': 0.2,
                           'axes.facecolor': '#EEEEEE',
                           'text.color': '#424242',
                           'legend.frameon': True})

def show_case_dtw(a,b):
    x = array(return_ratio.iloc[:,a]).reshape(-1, 1)
    y = array(return_ratio.iloc[:,b]).reshape(-1, 1)
    plt.figure(figsize=(10,5))
    plt.plot(return_ratio.index,x)
    plt.plot(return_ratio.index,y)
    plt.title('Future Return Rate', fontsize=22)
    plt.show()

    dist, cost, acc, path = dtw(x, y, dist=lambda x, y: norm(x - y, ord=1))
    # acc.T is the accumulated cost matrix returned from the dtw( ) function.
    # origin is which part of the chart to start the plot.
    # cmap is shorthand for color map. The 'prism' palette is particularly useful for visualizing DTW outputs.
    plt.figure(figsize=(10.3,10.3))
    plt.imshow(acc.T, origin='lower', cmap='tab20c', interpolation='nearest')
    plt.title('Cummulative Distance', fontsize=22)
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, acc.shape[0]-0.5))
    plt.ylim((-0.5, acc.shape[1]-0.5))
    plt.show()
    plt.clf() # Clear the current figure in-case we want to generate or try a different plot.
    print(dist)
    return None

''' Calculate DTW Matrix'''
n_col=np.shape(return_ratio)[1]
DTW_matrix=np.zeros((n_col,n_col))

for col in range(n_col):
    print(col)
    for row in range(n_col):
        if col>row:
            DTW_matrix[col,row]=DTW_matrix[row,col]
        else:
            x = array(return_ratio.iloc[:,col]).reshape(-1, 1)
            y = array(return_ratio.iloc[:,row]).reshape(-1, 1)
            distance, path = fastdtw(x, y, dist=lambda x, y: norm(x - y, ord=1))
            DTW_matrix[col,row]=distance