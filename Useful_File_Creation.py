# Creation of importat files used in the scripts
# This need to be done just once, if those file are not availabe or the atlas has been updated

from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
# read label file
from Readlabel import readlabel
  
path_files = Path('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Files')      

# Directory of the processed histology
processed_histology_folder = Path('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed')

# Paths of the atlas, segmentation and labels
## Atlas ##
atlas_folder = Path(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v2_pack')
atlas_path =  atlas_folder/'WHS_SD_rat_T2star_v1.01.nii.gz'
atlas = nib.load(atlas_path)
atlas_header = atlas.header
pixdim = atlas_header.get('pixdim')[1]
atlas_data = atlas.get_fdata()

## Mask ##
mask_folder = Path(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas')
mask_path = mask_folder/'WHS_SD_rat_brainmask_v1.01.nii.gz'
mask = nib.load(mask_path)
mask_data = mask.get_fdata()[:,:,:,0]
## Segmentation ##
segmentation_folder = Path(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas')
segmentation_path = segmentation_folder/'WHS_SD_rat_atlas_v4_beta.nii.gz'
segmentation = nib.load(segmentation_path)
segmentation_data = segmentation.get_fdata()
## Labels ##
labels_item = open(r"/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.label", "r")
labels_index, labels_name, labels_color, labels_initial = readlabel( labels_item ) 

### Atlas in RGB colors according with the label file ###
cv_plot = np.zeros(shape = (atlas_data.shape[0],atlas_data.shape[1],atlas_data.shape[2],3))
# here I create the array to plot the brain regions in the RGB
# of the label file
for i in range(len(labels_index)):
    coord = np.where(segmentation_data == labels_index[i][0])        
    cv_plot[coord[0],coord[1],coord[2],:] =  labels_color[i]
np.save(path_files/'cv_plot.npy', cv_plot)    
    
    
### Remove skull and tissues from atlas_data ###
CC = np.where(mask_data == 0)
atlas_data[CC] = 0
np.save(path_files/'atlas_data_masked.npy',atlas_data)

### Get the edges of the colors defined in the label ###
Edges = np.empty((512,1024,512))
for sl in range(0,1024):
    Edges[:,sl,:] = cv2.Canny(np.uint8((cv_plot[:,sl,:]*255).transpose((1,0,2))),100,200)  
np.save(path_files/'Edges.npy', Edges)



# here I create the array to plot the brain regions in the RGB
# of the label file
cv_plot_display = np.zeros(shape = (atlas_data.shape[0],atlas_data.shape[1],atlas_data.shape[2],3))
for i in range(len(labels_index)):
    if i == 0:
        coord = np.where(segmentation_data == labels_index[i][0])        
        cv_plot_display[coord[0],coord[1],coord[2],:] =  labels_color[i]        
    else:
        coord = np.where(segmentation_data == labels_index[i][0])        
        cv_plot_display[coord[0],coord[1],coord[2],:] =  [128,128,128]
np.save(path_files/'cv_plot_display.npy',cv_plot_display)        
