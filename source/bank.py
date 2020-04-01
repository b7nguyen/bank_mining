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
import seaborn as sns
import matplotlib.pyplot as plt
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
#from sklearn.tree.export import export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt


STATE_QUIT = -1
STATE_MAIN= 0
STATE_CLASSIFY = 1
STATE_VISUALIZE = 2




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
    
def decisionTree(data):
    
    cross_score = 0
    n_folds = 10
    
    target = data[TARGET_CLASS]
    data = data.drop([TARGET_CLASS], 1)
    data = oneHot(data)
    X_train, X_test, y_train, y_test = splitData(data, target, .3)
    
    
    
    dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=20)
    cross_score = cross_val_score(estimator=dt, 
                                  X=X_train, 
                                  y=y_train, 
                                  cv=n_folds
                                  )
    cross_score = round( (sum(cross_score[:])/cross_score.size) * 100, 2)
    
    clear_screen()
    print(f'{n_folds}-Cross Validation Score is {cross_score}\n')
    input('Press enter to continue')

    

#%%

def oneHot(data):
    #TBD

    obj_df = data.select_dtypes('object')
    data = pd.get_dummies(data, columns=obj_df.columns)
    
    return data

#%%

def showCategoricalData(df):
    #Extract the categorical variables to a new dataframe
    cat_var = df.select_dtypes(include=['object'])

    #Remove the Class attribute from the dataframe
    cat_var2 = cat_var.drop(['y'], axis = 1)

    #Create a Series with column headers
    cat_col= list(cat_var2.columns.values)
    cat_columns = pd.Series(cat_col)
    #Generate a 2 x 5 plot grid
    fig, axs = plt.subplots(2, 5, sharex=False, sharey=False, figsize=(20, 20))       

    counter = 0
    for x in cat_columns:
        col_name = x
        col_val = cat_var2[col_name].value_counts()
        #Define x,y position of each subplot for each column 
        plot_position_x = counter // 5
        plot_position_y = counter % 5
        x_pos = np.arange(len(col_val))
        axs[plot_position_x,plot_position_y].bar(x_pos,col_val.values,tick_label=col_val.index)
        axs[plot_position_x,plot_position_y].set_title(col_name)
        for tick in axs[plot_position_x,plot_position_y].get_xticklabels():tick.set_rotation(45)
        counter += 1
        
    
    plt.show()
    
#%%
def showHistograms(df):
    #Extract the numeric variables to a new dataframe
    num_var = df.select_dtypes(include=['number'])
    
    #Create a Series with column headers
    num_col= list(num_var.columns.values)
    num_series = pd.Series(num_col)
    
    #Generate a 2 x 5 plot grid
    fig, axs = plt.subplots(2, 5, sharex=False, sharey=False, figsize=(30, 10))       
    #Create histogram for each column
    counter = 0
    for x in num_series:
        col_name = x
        #Define x,y position of each subplot for each column 
        plot_position_x = counter // 5
        plot_position_y = counter % 5
        axs[plot_position_x,plot_position_y].hist(num_var[col_name])
        axs[plot_position_x,plot_position_y].set_title(col_name)
        
        counter += 1
        
    
    plt.show()

#%%



def showMainMenu(state):
    clear_screen()
    
    if (state == STATE_MAIN):
        print('a) Classify')
        print('b) Visualize')
        print('q) Quit')
               
        
    getInput = input('What would you like to do? ') 
    
    if (getInput == 'a'):
        state = STATE_CLASSIFY
    elif (getInput == 'b'):
        state = STATE_VISUALIZE
    elif(getInput == 'q'):
        state = STATE_QUIT
        
    return state
    
#%%

def showClassifyMenu(state, data):
    clear_screen()
    
    print('a) Decision Tree')
    print('b) Naive Bayes')
    print('q) Quit')
        
    getInput = input('What classifier would you like to model: ') 
    clear_screen()
    
    if(getInput.lower() == 'a'):
        decisionTree(data)
    
    
    state = STATE_MAIN
        
    return state
    
#%%

def showVisualizeMenu(state, data):
    clear_screen()
    df = data

    
    print('a) Visualize Nominal Frequency')
    print('b) Visualize Numeric Attribute Histograms')
    print('c) Show head of dataframe')
    print('q) Quit')
    
    getInput = input('How would you like to visualize the data? ')  
    
    clear_screen()
    
    if(getInput.lower() == 'a'):
        showCategoricalData(df)
    elif(getInput.lower() == 'b'):
        showHistograms(df)
    elif(getInput.lower() == 'c'):
        showHead(df)
        
    
    
    state = STATE_MAIN
    
    return state

#%%
def showHead(data):
    clear_screen()
    print(data.head)
    input('Press Enter to continue')



#%%
def clear_screen():
    #os.system("cls" if os.name == "nt" else "clear")
    print("\n"*100)

#%%



#All functions should go above this line
if __name__ == '__main__':
    X_train_ID = 0
    X_train = 0
    y_train = 0
    X_test_ID = 0
    X_test = 0
    y_test = 0
    target = 0    
    state = STATE_MAIN
    
    data = pd.DataFrame()
    

    '''Get the data from CSV file'''
    data = readCSVFile(FILETRAIN)
    
    clear_screen()

    while(True):
        
        if(state == STATE_MAIN):
            state = showMainMenu(state)
        if(state == STATE_CLASSIFY):
            state = showClassifyMenu(state, data)
        elif(state == STATE_VISUALIZE):
            state = showVisualizeMenu(state, data)
        else:
            break
            


    
    
    
   
    
    
    

    
    
    