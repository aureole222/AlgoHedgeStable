#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 22:50:13 2018
@author: yc
"""
#sudo pip install quandl
import quandl
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# IMPORT ALL THE PRODUCTS AND DATA ANALYSING
# =============================================================================
df = pd.read_csv('C:/Users/richard/Documents/stable/algorithm hedging/Book6.csv',
                   parse_dates=True, index_col='Date')
df.columns


def get_time_period(df):
    '''Get the shortest data and use that column as base track'''
    nulldata = df.isnull().sum()
    nulldata = nulldata.reset_index()
    nulldata.columns = ['col_names', 'null_count']
    sns.factorplot(x = 'null_count', y='col_names', data = nulldata, kind='bar')

#['2017-07-01':'2017-12-31']
def fill_na():
    global df
    dataset=df
    for ii,i in enumerate(dataset.index):
        for jj,j in enumerate(dataset):
            if str(dataset.iloc[ii,jj])=='nan':
                try:
                    if str(dataset.iloc[ii+1,jj])=='nan':
                        dataset.iloc[ii,jj]=dataset.iloc[ii-1,jj]
                    else:
                        dataset.iloc[ii,jj]=(dataset.iloc[ii-1,jj]+dataset.iloc[ii+1,jj])/2
                except:
                        dataset.iloc[ii,jj]=dataset.iloc[ii-1,jj]
    return dataset


get_time_period(df)
df = fill_na()
df.info()

# =============================================================================
# DATA ANALYSING
# =============================================================================
'''Find unusual value'''
def find_unusual_value():
    
    '''
    When drawing the plot, we can see
    there are some unsuitable in the data.
    So the function is to seek the problematic
    columns out
    '''
    find_zero=df.columns
    a=[]
    for value in range(len(df.columns)):
        a.append(any(df.iloc[:,value]==0))
    print(a)
    number = 0 
    for i in a:
        if i is False:
            number += 1
        else:
            break
    else:
        print('Nothing Bad!!')
    return find_zero[number]




'''Replace the 0 value'''


'''Select different time period'''
df.info()

'''Data Visualization'''
plt.figure(figsize=(40, 30))
correlation = df.corr()
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
sns.set(font_scale = 2)
sns.heatmap(correlation,annot=True,annot_kws={"size": 14},mask=mask)

'''Feature Scaling'''
scaler = StandardScaler()
standard_df = scaler.fit_transform(df.iloc[:,:])



'''PCA: Hedging Start'''
def pca_hedge(standard_df):
    global pca
    pca = PCA()
    standard_df = pca.fit_transform(standard_df)
    index_names = df.iloc[:,:len(df.columns)].columns
    col_names = ['PC' + str(x) for x in range(1,len(df.columns)+1)]
    PCA_Loadings = pd.DataFrame(pca.components_.T, index=index_names, columns=col_names)
    PCA_Loadings = PCA_Loadings.round(3)
    print([round(float(i),4) for i in pca.explained_variance_])#like the eigen value
    print([round(float(i),4) for i in pca.explained_variance_ratio_])


#The following code constructs the Screen plot
def PCA_importance():
    each_var = np.round(pca.explained_variance_ratio_* 100,2)
    labels = ['PC' + str(x) for x in range(1,len(df.columns)+1)]
    plt.figure(figsize=(15,11))
    plt.bar(range(1,len(each_var)+1),height=each_var, tick_label=labels,alpha=0.7)
    plt.ylabel('Percentage of Explained Variance %')
    plt.xlabel('Principal Component')
    plt.xticks(rotation=60)
    plt.show()

            
def select_product():
    each_ration = np.round(pca.explained_variance_ratio_* 100,2)
    pca_table = pd.DataFrame(pca.components_.T, 
                             columns=['PC' + str(x) for x in range(1,len(df.columns)+1)], 
                             index=df.columns)
#    for i in range(len(pca_table.columns)):
#        pca_table.iloc[:,i] = pca_table.iloc[:,i]*each_ration[i]
    pca_table = round(pca_table, 3)
    return pca_table
pca_hedge(standard_df)
PCA_importance()
pca_table = select_product()
print(pca_table)