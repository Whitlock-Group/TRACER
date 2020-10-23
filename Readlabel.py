"""
Created on Fri Oct  9 13:07:15 2020

@author: jacopop
"""

def readlabel( file ):
    labels = file.readlines()
    pure_labels = [ x for x in labels if "#" not in x ]
    
    return [pure_labels]
