#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:07:46 2020

@author: b7nguyen
"""

from datetime import datetime

import math
import random


from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import tree


#%%

X, y = load_iris(return_X_y = True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

#%%

tree.plot_tree(clf.fit(X, y)) 


a = input('press enter')