#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:27:37 2020

@author: jacopop
"""

import numpy as np

# class to read labels
def readlabel( file ):
    output_index = []
    output_names = []
    output_colors = []
    labels = file.readlines()    
    pure_labels = [ x for x in labels if "#" not in x ]
    
    for line in pure_labels:
        line_labels = line.split()
        accessed_mapping = map(line_labels.__getitem__, [0])
        L = list(accessed_mapping)
        indice = [int(i) for i in L] 
        accessed_mapping_rgb = map(line_labels.__getitem__, [1,2,3])
        L_rgb = list(accessed_mapping_rgb)
        colorsRGB = [int(i) for i in L_rgb]  
        output_colors.append(colorsRGB) 
        output_index.append(indice)
        output_names.append(' '.join(line_labels[7:]))         
    
    for i in range(len(output_names)):
        output_names[i] = output_names[i][1:-1]
        
    output_index = np.array(output_index)  # Use numpy array for  in the label file
    output_colors = np.array(output_colors)  # Use numpy array for  in the label file                  
    return [output_index, output_names, output_colors]
