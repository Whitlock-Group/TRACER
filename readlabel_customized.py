# Insert manually the rgb triplet for each area of the brain

from __future__ import print_function


import numpy as np



# Functions defined in separate files
# from Readlabel import readlabel

# class to read labels
def readlabel_c( file ):
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
        output_index.append(indice)
        output_names.append(' '.join(line_labels[7:]))         
    
    for i in range(len(output_names)):
        output_names[i] = output_names[i][1:-1]
        
    output_index = np.array(output_index)  # Use numpy array for  in the label fil


    label_colors_new = []
    for j in range(len(output_names)):
        print('\n'+output_names[j]+': \n')
        C = []
        # iterating till the range 
        for i in range(0, 3): 
            ele = int(input()) 
            C.append(ele) # adding the element 
        label_colors_new.append(C)
        
    output_colors = np.array(label_colors_new)  # Use numpy array for  in the label file         

    return [output_index, output_names, output_colors]