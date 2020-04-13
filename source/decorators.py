#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:45:48 2020

@author: b7nguyen
"""

import functools
import random
import os
 
#%%

def clear_screen():
    #os.system("cls" if os.name == "nt" else "clear")
    print("\n"*100)

#%%

def ml_init(func):
    """Notify that model is currently running """
    @functools.wraps(func)
    def wrapper_ml_init(*args, **kwargs):
        clear_screen()
        print("Creating ML model...")
        value = func(*args, **kwargs)
        clear_screen()
        
        return value
    
    
    return wrapper_ml_init






