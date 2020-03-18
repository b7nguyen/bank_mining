# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:59:05 2020

@author: ilear
"""

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pathlib
    
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree.export import export_text
from sklearn.preprocessing import LabelEncoder




PATH = "C:/Users/ilear/Documents/Coding/Data Mining/Data mining practicum/Assignment 4/My datasets/"
FILETRAIN = "bank-additional-full-sep.csv"
FILETEST = "bank-additional-full-sep.csv"
TARGET_CLASS = "y"
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



if __name__ == '__main__':
    X_test_ID = 0
    X_test = 0
    y_test = 0
    train_target = 0
    test_target = 0
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    
    
    #Get the data from CSV file
    data_train = readCSVFile(FILETRAIN)
    data_test = readCSVFile(FILETEST)
    
    target = 0    
    data = pd.DataFrame()
    
    
    '''Get the data from CSV file'''
    data = readCSVFile(FILETRAIN)
print (data.head(n=5))


'''
    Set the data to be trained and target class 
    
'''
    
target = data[TARGET_CLASS]
data = data.drop([TARGET_CLASS], 1)
    

'''
@ -68,13 +115,23 @@ if __name__ == '__main__':
    seperate test set 
'''
    
X_train, X_test, y_train, y_test = splitData(data, target, .3)
    
    
'''
    One hot encoder
'''
data = oneHot(data)
    
print (data.head(n=5))
    
'''
    Train and evaluate decision tree 
    cross_score = bn_decisionTree(trainFrame, y_train)
    cross_score = decisionTree(trainFrame, y_train)
    '''
    

decisionTree(data, target)
    
    
    
