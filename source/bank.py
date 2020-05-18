#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:30:59 2020

@author: b7nguyen
"""


"""
Get Library
"""

import numpy as np
import os
'''import graphviz'''
import pathlib



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
#from sklearn.tree.export import export_text

from decorators import ml_init

import seaborn as sns
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import pandas as pd
from pandas.api.types import is_numeric_dtype

from imblearn.over_sampling import SMOTENC

from bank_class import MyFrame

STATE_QUIT = -1
STATE_MAIN= 0
STATE_CLASSIFY = 1
STATE_VISUALIZE = 2
STATE_PREPROCESS = 3




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
#Renames specific column
def rename(DFmain, current_col, new_col):
    data  = DFmain.dframe
    data = data.rename(columns = {current_col,new_col})
    return data
#%%
def reshape_data(DFmain):
    data = DFmain.dframe
    dataset = data.values
    #Defining X as all columns except the last column
    X = dataset[:, :-1]
    
    #Defining y as all rows in last column
    y = dataset[:,-1]
    
    #Restructuring y to be a column
    y = y.reshape((len(y),1))
    return X,y

#%%
#NEEDS to be generalized- categorical feature indices specifically
def SMOTE_cat(DFmain):
    data = DFmain.dframe
    
    #Create list of all column names
    list_col = list(data.columns)
    
    #Create list of column names in X_train
    x_train_col = list_col[0:(len(list_col)-1)]
    
    X, y = reshape_data(DFmain)
    
    X_train, X_test, y_train, y_test = splitData(X,y, test_size= .33)
        
    sm = SMOTENC(categorical_features=[1,2,3,4,5,6,7,8,9,14],random_state= 1,
             sampling_strategy ='minority') 
    X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train.ravel())
    
    
    print("Before SMOTE, counts of label 'yes': {}".format(sum(y_train 
                                                                 == 'yes')))
    print("After SMOTE, the shape of X_train: ", X_train_smote.shape) 
    print("After SMOTE, the shape of y_train: ", y_train_smote.shape)  
    print("After SMOTE, counts of Class attr 'Yes': ", sum(y_train_smote 
                                                           == 'yes'))
    print("After SMOTE, counts of Class attr 'No': ", sum(y_train_smote 
                                                          == 'no'))
    #Convert SMOTE processed ndarrays back into a dataframe
    SMOTE_train_df = pd.DataFrame(data= X_train_smote, index= None, columns = 
                          x_train_col )
    #Convert Y training set into a Series and add on to dataframe
    Y_train_series= pd.Series(y_train_smote)
    
    SMOTE_train_df = SMOTE_train_df.assign(y = Y_train_series)
    
    
    print('\n\na) Go back to main menu')
    print('b) Go back to pre-processing menu')
    print('q) Quit')
    
    getInput = input('What would you like to do next: ')  
    
    if(getInput.lower() == 'a'):
        state = STATE_MAIN
    elif(getInput.lower() == 'b'):
        state = STATE_PREPROCESS
        showPreProcessMenu(state,data)
        
    return state, SMOTE_train_df
        
#%%
   
@ml_init
def ML_decisionTree(DFmain):
    
    cross_score = 0
    n_folds = 10
    data = DFmain.dframe
    
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
    
    #cross_score is an array of 10 scores. To get the score, we find the avg. 
    cross_score = round( (sum(cross_score[:])/cross_score.size) * 100, 2)
    
    clear_screen()
    print(f'{n_folds}-Cross Validation Score is {cross_score}\n')
    input('Press enter to continue')


#%%



@ml_init
def ML_naiveBayes(DFmain):
    data = DFmain.dframe
    gnb = GaussianNB()
    score = 0 
    target = data[TARGET_CLASS]
    
    data = data.drop([TARGET_CLASS], 1)
    data = oneHot(data)  
    X_train, X_test, y_train, y_test = splitData(data, target, .3)
    
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    
    gnb.fit(data, target)
    y_pred = gnb.predict(data)
    
    score = round(metrics.accuracy_score(target, y_pred) * 100, 2)
    
    
    print(f'Naive Bayes score is {score}\n')
    input('Press enter to continue')  


#%%

'''
Input: Input Dataframe with the class attribute removed. Ok to leave numeric 
    Attributes in
Output: Returns Dataframe with all nominal attributes encoded to binary values. 
'''
def oneHot(data):

    obj_df = data.select_dtypes('object')
    data = pd.get_dummies(data, columns=obj_df.columns)
    #print(data.head())
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
    row_count, col_count = createSubPlots(cat_col)
    #Generate a 2 x 5 plot grid
    fig, axs = plt.subplots(row_count, col_count, sharex=False, sharey=False, figsize=(20, 20))       

    counter = 0
    
    #Create a frequency graph for each nominal attribute column 
    for x in cat_columns:
        col_name = x
        col_val = cat_var2[col_name].value_counts()
        
        #Define x,y position of each subplot for each column 
        plot_position_x = counter // col_count
        plot_position_y = counter % col_count
        x_pos = np.arange(len(col_val))
        
        #Create frequency graph of each unique val in the column
        axs[plot_position_x,plot_position_y].bar(x_pos,col_val.values,
                                                 tick_label=col_val.index)
        axs[plot_position_x,plot_position_y].set_title(col_name)
        for tick in axs[plot_position_x,plot_position_y].get_xticklabels():
            tick.set_rotation(45)
        counter += 1
        
    
    plt.show()
    
#%%
def showScatterPlot(df):
    #Extract numeric variables to a new dataframe
    num_var = df.select.dtypes(includes=['number'])
    
    #Create a Series with column headers
    num_col= list(num_var.columns.values)
    num_series = pd.Series(num_col)
    #MK: Please expalin this code
    
    #Generate a 2 x 5 plot grid
    fig, axs = plt.subplots(5, 5, sharex=False, sharey=False, figsize=(30, 10))
    
    #Create a Scatterplot for each column (I think?)
    counter = 0
    for x in num_series:
        col_name = x
        #Define x,y position of each subplot for each column 
        plot_position_x = counter // 5
        plot_position_y = counter % 5
        axs[plot_position_x,plot_position_y].sns.pairplot(num_var[col_name])
        axs[plot_position_x,plot_position_y].set_title(col_name)
        
        counter += 1
        
    
    plt.show()
   
    
    
#%%
def showHistograms(df):
    #Extract the numeric variables to a new dataframe
    num_var = df.select_dtypes(include=['number'])
    
    #Create a Series with column headers
    num_col= list(num_var.columns.values)
    num_series = pd.Series(num_col)
    
    row_count, col_count = createSubPlots(num_col)
    #Generate a 2 x 5 plot grid
    fig, axs = plt.subplots(row_count, col_count, sharex=False, sharey=False, figsize=(30, 10))       
    
    #Create histogram for each numeric attribute column
    counter = 0
    for x in num_series:
        col_name = x
        
        #Define x,y position of each subplot for each column 
        plot_position_x = counter // col_count
        plot_position_y = counter % col_count
        
        #Create histogram for the column 
        axs[plot_position_x,plot_position_y].hist(num_var[col_name])
        axs[plot_position_x,plot_position_y].set_title(col_name)
        
        counter += 1
        
    
    plt.show()

#%%
def createSubPlots(list):
    count_num_list=len(list)
    
    #Define the number of rows and columns for the subplots
    if count_num_list %2 == 0:
        col_count= int(count_num_list / 2)
        row_count = int(count_num_list / col_count)
        
    else: 
        make_even =count_num_list +1
        col_count= int(make_even / 2)
        row_count = int(make_even / col_count)
    
    return row_count, col_count
#%%

@ml_init
def showBoxPlots(df):
    #Calculate how many subplots are needed
    num_col_list = list(df.select_dtypes(include=[np.number]).columns.values)     
    row_count, col_count = createSubPlots(num_col_list)
    
    #Create the subplots
    fig, axes = plt.subplots(row_count, col_count, sharex=False, sharey=False, figsize=(20, 20))       
        
    #Create a boxplot in each subplot for each numeric value column
    counter = 0
    
    for column in df:
        if is_numeric_dtype(df[column]) is True:
            plot_position_x = counter // col_count
            plot_position_y = counter % col_count
            sns.boxplot(x="y", y= column, data=df, ax= axes[plot_position_x,plot_position_y])
            print (column)
            print(df.groupby(["y"])[column].describe()) 
            counter += 1
            
        else:
            continue 
    plt.show()
    
    input('Press enter to continue')

#%%

def showMainMenu(DFmain, state):
    clear_screen()
    
    if (state == STATE_MAIN):
        print('Machine Learning using Classification on a Bank Marketing Dataset')
        print('Business Objective: Predict the outcome of a marketing campaign from customers attributes')
        print('a) Classify')
        print('b) Visualize')
        print('c) Pre-Process Data')
        print('q) Quit')
               
        
    getInput = input('What would you like to do? ') 
    
    if (getInput == 'a'):
        state = STATE_CLASSIFY
    elif (getInput == 'b'):
        state = STATE_VISUALIZE
    elif (getInput == 'c'):
        state = STATE_PREPROCESS
    elif(getInput == 'q'):
        state = STATE_QUIT
        
    return state
    
#%%


def showClassifyMenu(DFmain, state):
    clear_screen()
    
    print('a) Decision Tree')
    print('b) Naive Bayes')
    print('q) Quit')
        
    getInput = input('What classifier would you like to model: ') 
    
    if(getInput.lower() == 'a'):
        ML_decisionTree(DFmain)
    elif(getInput.lower() == 'b'):
        ML_naiveBayes(DFmain)
    
    
    state = STATE_MAIN
        
    return state
    
#%%

def showVisualizeMenu(state, data):
    clear_screen()
    df = data

    
    print('a) Visualize Nominal Frequency')
    print('b) Visualize Numeric Attribute Histograms')
    print('c) Visualize Boxplots')
    print('d) Show head of dataframe')
    print('q) Quit')
    
    getInput = input('How would you like to visualize the data? ')  
    
    clear_screen()
    
    if(getInput.lower() == 'a'):
        showCategoricalData(df)
    elif(getInput.lower() == 'b'):
        showHistograms(df)
    elif(getInput.lower() == 'c'):
        showBoxPlots(df)
    elif(getInput.lower() == 'd'):
        showHead(df)    
    
    
    state = STATE_MAIN
    
    return state
#%%
def showPreProcessMenu(DFmain, state):
    data = DFmain.dframe
    clear_screen()
    
        
    print('a) Balance the dataset with categorical values using SMOTENC')
    print('b) One-hot encode all columns')
    print('q) Quit')
    
    getInput = input('How would you like to pre-process the data? ')  
    
    clear_screen()
    
    if(getInput.lower() == 'a'):
        SMOTE_cat(DFmain)
    elif(getInput.lower() == 'b'):
        print ('Building in Progress')
        

    
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

    
    data = pd.DataFrame()
    
    

    '''Get the data from CSV file'''
    data = readCSVFile(FILETRAIN)
    DFmain = MyFrame(data)

    
    state = STATE_MAIN
    clear_screen()

    while(True):
        
        if(state == STATE_MAIN):
            state = showMainMenu(DFmain, state)
        elif(state == STATE_CLASSIFY):
            state = showClassifyMenu(DFmain, state)
        elif(state == STATE_VISUALIZE):
            state = showVisualizeMenu(state, data)
        elif(state == STATE_PREPROCESS):
             state = showPreProcessMenu(DFmain, state)
        else:
            break
            


  
    
    