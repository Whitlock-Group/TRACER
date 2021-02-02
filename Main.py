# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:51:13 2020

@author: jacopop
"""
# Import libraries
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Functions defined in separate files
from Readlabel import readlabel


# Directory of histology
histology_folder = r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Matlab\preprocessed\processed'


# Insert the plane of interest
plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')
# Check if the input is correct
while plane.lower() != 'c' and plane.lower() != 's' and plane.lower() != 'h':
    print('Error: Wrong plane name')
    plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')

# Paths of the atlas, segmentation and labels
# Atlas
atlas_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_T2star_v1.01.nii.gz')
# Segmentation
segmentation_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_atlas_v3.nii.gz')
# Labels
labels_item = open(r"C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v3.label", "r")
labels = readlabel( labels_item )

# Load the atlas and segmentation
atlas = nib.load(atlas_path)
data_atlas = atlas.get_fdata()
segmentation = nib.load(atlas_path)
data_segmentation = segmentation.get_fdata()

# Select plane for visualization
if plane.lower() == 'c':
    av_plot = data_atlas 
    tv_plot = data_segmentation
elif plane.lower() == 's':
    av_plot = data_atlas.transpose(2,1,0)
    tv_plot = data_segmentation.transpose(2,1,0)
elif plane.lower() == 'h':
    av_plot = data_atlas.transpose(1,2,0)
    tv_plot = data_segmentation.transpose(1,2,0)

# Visualize the slice of interest for the histology
# (maybe I need a separate function for this)

reference_shape = tv_plot.shape;

histology = Image.open(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Matlab\preprocessed\aaa.tif')
histology.show()
histology_array = np.array(histology)


# Interact with the figure





# plot the slices of the atlas
plt.imshow(av_plot[:,:,300], cmap='gray',)
plt.plot()

plt.imshow(tv_plot[:,:,300])
plt.plot()


#fig = plt.figure()
#ax = fig.gca(projection='3d')

#ax.voxels(av_plot, edgecolor="k")

#plt.show()
