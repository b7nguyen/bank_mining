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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import graphviz


PATH = "../input"
FILETRAIN = "/bank-additional-full.csv"
FILETEST = "/bank-additional-full.csv"

#%%
if __name__ == '__main__':
    X_train_ID = 0
    X_train = 0
    y_train = 0
    X_test_ID = 0
    X_test = 0
    y_test = 0
    train_target = 0
    test_target = 0
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    
    
    #Get the data from CSV file
    data_train = bn_readCSVFile(FILETRAIN)
    data_test = bn_readCSVFile(FILETEST)
    
    
    '''
    Set the data to be trained and target class 
    
    '''
    
    #trainFrame = data_train[['Pclass', 'SibSp', 'Parch', 'Fare']]
    #testFrame = data_test[['Pclass', 'SibSp', 'Parch', 'Fare']]
    
    

    '''
    Split the data to be trained and tested. This is not required if given a 
    seperate test set 
    '''
    
    
    '''
    Train and evaluate decision tree 
    cross_score = bn_decisionTree(trainFrame, y_train)
    '''
    

    
    

    
    
    
    #X_train_ID, X_train, y_train, X_test_ID, X_test = setFrame(test, train)
    
    
    
   
    
    
    

    
    
    
    
    #modelLR(test, train)
    
    
    