#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:45:35 2020

@author: b7nguyen
"""

import pandas as pd
import numpy as np

class MyFrame(object):
    def __init__(self, dframe):
        self.dframe = dframe
        
        
    def save(self, dframe):
        self.dframe = dframe
        
    