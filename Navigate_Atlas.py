# First run the nex line in the console
# %matplotlib qt5
# Import libraries
from __future__ import print_function

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2

# Functions defined in separate files
from Readlabel import readlabel

# Class to scroll the atlas slices
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Atlas viewer')
        print('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2        
        
        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        #ndimage.rotate(img, 45, reshape=False)
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        

# Directory of the processed histology
processed_histology_folder = r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\histology\processed'
file_name = input('Histology file name: ')
histology = Image.open(os.path.join(processed_histology_folder, file_name+'_processed.tif')).copy()

# Insert the plane of interest
plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')
# Check if the input is correct
while plane.lower() != 'c' and plane.lower() != 's' and plane.lower() != 'h':
    print('Error: Wrong plane name')
    plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')

# Paths of the atlas, segmentation and labels
# Atlas
atlas_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_T2star_v1.01.nii.gz')
# Mask
mask_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_brainmask_v1.01.nii.gz')
# Segmentation
segmentation_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_atlas_v3.nii.gz')
# Labels
labels_item = open(r"C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v3.label", "r")
labels = readlabel( labels_item )

print('ok')

# Load the atlas, mask, and segmentation
atlas = nib.load(atlas_path)
data_atlas = atlas.get_fdata()
mask = nib.load(mask_path)
mask_data_intermediate = mask.get_fdata()
mask_data = mask_data_intermediate[:,:,:,0]
segmentation = nib.load(atlas_path)
data_segmentation = segmentation.get_fdata()
print('ok')

# Select plane for visualization and rotate the atles
if plane.lower() == 'c':
    d1 = 512
    d2 = 512
    d3 = 1024
    av_plot = np.zeros(shape = (d1,d2,d3))
    tv_plot = np.zeros(shape = (d1,d2,d3))
    av_plot_temp = data_atlas.transpose((0,2,1))
    tv_plot_temp = data_segmentation.transpose((0,2,1))  
    #mask_data_t = mask_data.transpose((0,2,1))
    #cc = np.where(mask_data_t == 0)
    #av_plot_temp[cc] = 0 
    print('ok')
    for i in range(av_plot_temp.shape[2]):
        av_plot[:, :, i] = ndimage.rotate(av_plot_temp[:, :, i], 90)    
        tv_plot[:, :, i] = ndimage.rotate(tv_plot_temp[:, :, i], 90)    
elif plane.lower() == 's':
    d1 = 512
    d2 = 1024
    d3 = 512
    av_plot = np.zeros(shape = (d1,d2,d3))
    tv_plot = np.zeros(shape = (d1,d2,d3))
    av_plot_temp = data_atlas.transpose((1,2,0))
    tv_plot_temp = data_segmentation.transpose((1,2,0))
    mask_data_t = mask_data.transpose((1,2,0))
    cc = np.where(mask_data_t == 0)
    av_plot_temp[cc] = 0    
    print('ok')
    for i in range(av_plot_temp.shape[2]):
        av_plot[:, :, i] = ndimage.rotate(av_plot_temp[:, :, i], 90)    
        tv_plot[:, :, i] = ndimage.rotate(tv_plot_temp[:, :, i], 90)        
elif plane.lower() == 'h':
    d1 = 1024
    d2 = 512
    d3 = 512
    av_plot = np.zeros(shape = (d1,d2,d3))
    tv_plot = np.zeros(shape = (d1,d2,d3))
    av_plot_temp = data_atlas
    tv_plot_temp = data_segmentation    
    mask_data_t = mask_data
    cc = np.where(mask_data_t == 0)
    av_plot_temp[cc] = 0
    for i in range(av_plot_temp.shape[2]):
        av_plot[:, :, i] = ndimage.rotate(av_plot_temp[:, :, i], 90)    
        tv_plot[:, :, i] = ndimage.rotate(tv_plot_temp[:, :, i], 90)
        
# Show the Atlas        
my_dpi = 39
fig, ax = plt.subplots(1, 1, figsize=(float(av_plot.shape[0])/my_dpi,float(av_plot.shape[1])/my_dpi))
tracker = IndexTracker(ax, av_plot)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()
# Fix size and location of the figure window
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(800,300,d1,d2)


# Show the processed histology
my_dpi = 39
# Set up figure
fig = plt.figure(figsize=(float(histology.size[0])/my_dpi,float(histology.size[1])/my_dpi),dpi=my_dpi)
ax = fig.add_subplot(111)
# Remove whitespace from around the image
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set_title("Slice viewer")
# Show the histology image  
ax.imshow(histology)
plt.show()
# Fix size and location of the figure window
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(150,300,d1,d2)

# Plot the atlas 3D
# and plot everything

#fig_3d = plt.figure()
#ax_3d = fig_3d.gca(projection='3d')
#ax_3d.plot(data_atlas)

#plt.show()


 