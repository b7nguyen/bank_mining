#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:30:59 2020

@author: b7nguyen
"""


"""
Get Library
"""
import pandas as pd
import numpy as np
import os
'''import graphviz'''
import pathlib
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
#from sklearn.tree.export import export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




PATH = "../input"
FILETRAIN = "/bank-additional-full.csv"
FILETEST = "/bank-additional-full.csv"
TARGET_CLASS = 'y'
#%%
'''All functions should go below this line'''

def splitData(data, target, test_size):
      
    a_train, b_train, a_test, b_test = train_test_split(data, 
                                                        target, 
                                                        test_size = test_size,
                                                        random_state = 10
                                                        )
    
    return (a_train, b_train, a_test, b_test)
    
#%%


def readCSVFile(filename):
    
    file = PATH + filename
    return (pd.read_csv(file))




#%%
    
def decisionTree(X_train, y_train):
    
    cross_score = 0
    n_folds = 10
    
    dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=20)
    cross_score = cross_val_score(estimator=dt, 
                                  X=X_train, 
                                  y=y_train, 
                                  cv=n_folds
                                  )
    cross_score = round( (sum(cross_score[:])/cross_score.size) * 100, 2)
    print(f'{n_folds}-Cross Validation Score is {cross_score}')

    

#%%

def oneHot(data):
    #TBD
    
    obj_df = data.select_dtypes('object')
    data = pd.get_dummies(data, columns=obj_df.columns)
    
    return data



#%%
'''All functions should go above this line'''


if __name__ == '__main__':
    X_train_ID = 0
    X_train = 0
    y_train = 0
    X_test_ID = 0
    X_test = 0
    y_test = 0
    target = 0    
    data = pd.DataFrame()
    
    
    '''Get the data from CSV file'''
    data = readCSVFile(FILETRAIN)

    print (data.info())

    ''' Convert data into dataframe'''
    df = pd.DataFrame(data)
    
    '''show the first five rows'''
    
    print (df.head)

    ''' Examine number of unique values in each column'''

    print(df.nunique(axis=0,dropna=True))


    '''
    Set the data to be trained and target class 
    '''
    
    target = data[TARGET_CLASS]
    data = data.drop([TARGET_CLASS], 1)
    

    '''
    Split the data to be trained and tested. This is not required if given a 
    seperate test set 
    '''
    
    X_train, X_test, y_train, y_test = splitData(data, target, .3)
    
    
    '''
    One hot encoder
    '''
    data = oneHot(data)
    
    
    
    '''
    Train and evaluate decision tree 
    cross_score = decisionTree(trainFrame, y_train)
    '''
    
    decisionTree(data, target)
    
    
    

    
    
    
    #X_train_ID, X_train, y_train, X_test_ID, X_test = setFrame(test, train)
    
    
    
   
    
    
    

    
    
    
    
    #modelLR(test, train)
    
    
    