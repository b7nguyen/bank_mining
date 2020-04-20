#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:07:46 2020

@author: b7nguyen
"""

from datetime import datetime
import pandas as pd

import math
import random

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import tree


#%%
df = pd.DataFrame([['bar', 'one'], ['bar', 'two'],
                  ['foo', 'one'], ['foo', 'two']],
                  columns=['first', 'second'])

