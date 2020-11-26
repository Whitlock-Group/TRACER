# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:33:18 2020

@author: jacopop
"""


from __future__ import print_function

# First run the nex line in the console
# %matplotlib qt5
# Import libraries
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import math 
import mplcursors
from nilearn.image import resample_img
from nibabel.affines import apply_affine
import keyboard
from skimage import io, transform

# Functions defined in separate files
# from Readlabel import readlabel

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

# Classes to scroll the atlas slices (coronal, sagittal and horizontal)
class IndexTracker_c(object):
    def __init__(self, ax, X, pixdim):
        self.ax = ax
        ax.set_title('Atlas viewer')
        print('use scroll wheel to navigate the atlas \n')

        self.X = X
        if len(self.X.shape) == 3:
            rows, self.slices, cols = X.shape
            self.ind = 540
            self.im = ax.imshow(self.X[:, self.ind, :].T, origin="lower", extent=[0, 512*pixdim, 0, 512*pixdim], cmap='gray')
        elif len(self.X.shape) == 4:
            rows, self.slices, cols, color = X.shape
            self.ind = 653              
            self.im = ax.imshow(self.X[:, self.ind, :].transpose((1,0,2)), origin="lower",alpha=0.5)
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if len(self.X.shape) == 3:
            self.im.set_data(self.X[:, self.ind, :].T)
        elif len(self.X.shape) == 4:             
            self.im.set_data(self.X[:, self.ind, :].transpose((1,0,2)))
        #ndimage.rotate(img, 45, reshape=False)
        ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()  
        
class IndexTracker_s(object):
    def __init__(self, ax, X, pixdim):
        self.ax = ax
        ax.set_title('Atlas viewer')
        print('use scroll wheel to navigate the atlas \n')

        self.X = X
        if len(self.X.shape) == 3:
            self.slices, rows, cols = X.shape
            self.ind = 246                
            self.im = ax.imshow(self.X[self.ind, :, :].T, origin="lower", extent=[0 ,1024*pixdim, 0, 512*pixdim], cmap='gray')
        elif len(self.X.shape) == 4:                 
            self.slices, rows, cols, color = X.shape
            self.ind = 246                
            self.im = ax.imshow(self.X[self.ind, :, :].transpose((1,0,2)), origin="lower",alpha=0.5)
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if len(self.X.shape) == 3:
            self.im.set_data(self.X[self.ind, :, :].T)
        elif len(self.X.shape) == 4:             
            self.im.set_data(self.X[self.ind, :, :].transpose((1,0,2)))
        #ndimage.rotate(img, 45, reshape=False)
        ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()               
        
class IndexTracker_h(object):
    def __init__(self, ax, X, pixdim):
        self.ax = ax
        ax.set_title('Atlas viewer')
        print('use scroll wheel to navigate the atlas \n')

        self.X = X
        if len(self.X.shape) == 3:
            rows, cols, self.slices = X.shape
            self.ind = 440       
            self.im = ax.imshow(self.X[:, :, self.ind].T, origin="lower", extent=[0, 512*pixdim, 0, 1024*pixdim], cmap='gray')
        elif len(self.X.shape) == 4:                 
            rows, cols, self.slices, color = X.shape
            self.ind = 440         
            self.im = ax.imshow(self.X[:, :, self.ind].transpose((1,0,2)), origin="lower",alpha=0.5)
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()        
        
    def update(self):
        if len(self.X.shape) == 3:
            self.im.set_data(self.X[:, :, self.ind].T)
        elif len(self.X.shape) == 4: 
            self.im.set_data(self.X[:, :, self.ind].transpose((1,0,2)))
        #ndimage.rotate(img, 45, reshape=False)
        ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()   
        
class IndexTracker_cg(object):
    def __init__(self, ax, X, pixdim, S):
        self.ax = ax
        ax.set_title('Atlas viewer')
        print('use scroll wheel to navigate the atlas \n')
        
        self.X = X
        rows, self.slices, cols = X.shape
        self.ind = S
        self.im = ax.imshow(self.X[:, self.ind, :], origin="lower", alpha=0.5, extent=[0, 512*pixdim, 512*pixdim, 0], cmap='gray')
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()        
        
    def update(self):
        self.im.set_data(self.X[:, self.ind, :])
        #ndimage.rotate(img, 45, reshape=False)
        ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()      
        
        
        

# Directory of the processed histology
# for mac user 
#processed_histology_folder = r'/Users/jacopop/Box\ Sync/macbook/Documents/KAVLI\histology/processed'
# For windows users
processed_histology_folder = r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\histology\processed'
# for mac user 
# histology = Image.open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/rat_processed.tif').copy()
# For windows users
file_name = input('Histology file name: ')
# Mac
#img_hist = cv2.imread(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/rat_processed.tif',cv2.IMREAD_GRAYSCALE)
# Windows
img_hist_temp = Image.open(os.path.join(processed_histology_folder, file_name+'_processed.jpeg')).copy()
dpi_hist = img_hist_temp.info['dpi'][1]
pixdim_hist = 25.4/dpi_hist # 1 inch = 25,4 mm
img_hist = cv2.imread(os.path.join(processed_histology_folder, file_name+'_processed.jpeg'),cv2.IMREAD_GRAYSCALE)
# Insert the plane of interest
plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')
# Check if the input is correct
while plane.lower() != 'c' and plane.lower() != 's' and plane.lower() != 'h':
    print('Error: Wrong plane name')
    plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')

# Paths of the atlas, segmentation and labels
# Atlas
atlas_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v2_pack', 'WHS_SD_rat_T2star_v1.01.nii.gz')
# Mask
mask_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_brainmask_v1.01.nii.gz')
# Segmentation
segmentation_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_atlas_v4_beta.nii.gz')
# Labels
# Mac
#labels_item = open(r"/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.label", "r")
# Windows
labels_item = open(r"C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v4_beta.label", "r")
#labels = readlabel( labels_item )
labels_index, labels_name, labels_color = readlabel( labels_item )
    
# Load the atlas, mask, color and segmentation
# Mac
#atlas = nib.load(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v2_pack/WHS_SD_rat_T2star_v1.01.nii.gz')
# Windows
atlas = nib.load(atlas_path)
atlas_header = atlas.header
pixdim = atlas_header.get('pixdim')[1]
#atlas_data = atlas.get_fdata()
#atlas_affine = atlas.affine
atlas_data = np.load('atlas_data_masked.npy')
# Mac
#mask = nib.load(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_brainmask_v1.01.nii.gz')
# Windows
#mask = nib.load(mask_path)
#mask_data = mask.get_fdata()[:,:,:,0]
# Mac
#segmentation = nib.load('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.nii.gz')
# Windows
segmentation = nib.load(segmentation_path)
segmentation_data = segmentation.get_fdata()

# Atlas in RGB colors according with the label file
cv_plot = np.load('cv_plot.npy')/255
#cv_plot = np.zeros(shape = (atlas_data.shape[0],atlas_data.shape[1],atlas_data.shape[2],3))
# here I create the array to plot the brain regions in the RGB
# of the label file
#for i in range(len(labels_index)):
#    coord = np.where(segmentation_data == labels_index[i][0])        
#    cv_plot[coord[0],coord[1],coord[2],:] =  labels_color[i]

##################
# Remove skull and tissues from atlas_data
# CC = np.where(mask_data == 0)
# atlas_data[CC] = 0
##################

# Display the atlas
# resolution
dpi_atl = 25.4/pixdim
# Bregma coordinates
textstr = 'Bregma (mm): c = %.3f, h = %.3f, s = %.3f \nBregma (voxels): c = 653, h = 440, s = 246' %( 653*pixdim, 440*pixdim, 246*pixdim)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
if plane.lower() == 'c':
    # dimensions
    d1 = 512
    d2 = 512
    d3 = 1024
    fig, ax = plt.subplots(1, 1)#, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl), dpi=dpi_atl)
    tracker = IndexTracker_c(ax, atlas_data, pixdim)
    #tracker = IndexTracker_c(ax, cv_plot)
    #tracker_overlay = IndexTracker_c(ax, atlas_data)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    #fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # place a text box with bregma coordinates 
    # in bottom left in axes coords
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
    # display the coordinates relative to the bregma
    # when hovering with the cursor
    def format_coord(x, y):
        AP = tracker.ind*pixdim - 653*pixdim
        ML = y - 246*pixdim
        Z = x - 440*pixdim
        if ML >0:        
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, ML, Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, ML, Z)    
    ax.format_coord = format_coord
    plt.show()
# =============================================================================
#     cursor = mplcursors.cursor(hover=True)
#     # Show the names of the regions
#     def show_annotation(sel):
#         xi, yi = sel.target/pixdim
#         if np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1)).size:
#             Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1))[0,0]]
#         else:
#             # display nothing
#             Text = ' '
#         sel.annotation.set_text(Text)
#     cursor.connect('add', show_annotation)            
# =============================================================================
elif plane.lower() == 's':
    # dimensions
    d1 = 1024 
    d2 = 512    
    d3 = 512
    fig, ax = plt.subplots(1, 1, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl))
    tracker = IndexTracker_s(ax, atlas_data, pixdim)
    #tracker = IndexTracker_s(ax, cv_plot)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # place a text box with bregma coordinates 
    # in bottom left in axes coords
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
    # display the coordinates relative to the bregma
    # when hovering with the cursor
    def format_coord(x, y):
        AP = y - 653*pixdim
        ML = tracker.ind*pixdim - 246*pixdim
        Z = x - 440*pixdim
        if ML >0:        
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, ML, Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, ML, Z)    
    ax.format_coord = format_coord
    plt.show()
# =============================================================================
#     cursor = mplcursors.cursor(hover=True)
#     # Show the names of the regions 
#     def show_annotation(sel):
#         xi, yi = sel.target/pixdim
#         if np.argwhere(np.all(labels_index == segmentation_data[tracker.ind,int(math.modf(xi)[1]),int(math.modf(yi)[1])], axis = 1)).size:
#             Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[tracker.ind,int(math.modf(xi)[1]),int(math.modf(yi)[1])], axis = 1))[0,0]]
#         else:
#             # display nothing
#             Text = ' '
#         sel.annotation.set_text(Text)  
#     cursor.connect('add', show_annotation)            
# =============================================================================
elif plane.lower() == 'h':
    # dimensions
    d1 = 512
    d2 = 1024
    d3 = 512    
    fig, ax = plt.subplots(1, 1, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl))
    tracker = IndexTracker_h(ax, atlas_data, pixdim)
    #tracker = IndexTracker_h(ax, cv_plot)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # place a text box with bregma coordinates 
    # in bottom left in axes coords
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
    # display the coordinates relative to the bregma
    # when hovering with the cursor
    def format_coord(x, y):
        AP = x - 653*pixdim
        ML = y - 246*pixdim        
        Z = tracker.ind*pixdim - 440*pixdim
        if ML >0:        
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, ML, Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, ML, Z)    
    ax.format_coord = format_coord
    plt.show()
# =============================================================================
#     cursor = mplcursors.cursor(hover=True)
#     # Show the names of the regions
#     def show_annotation(sel):
#         xi, yi = sel.target/pixdim
#         if np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),int(math.modf(yi)[1]),tracker.ind], axis = 1)).size:
#             Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),int(math.modf(yi)[1]),tracker.ind], axis = 1))[0,0]]
#         else:
#             # display nothing
#             Text = ' '
#         sel.annotation.set_text(Text)
#     cursor.connect('add', show_annotation)
# =============================================================================
# Fix size and location of the figure window
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(800,300,d2,d1)     

# Store the current slice
S = tracker.ind 


# Show the processed histology
# Set up figure
fig_hist, ax_hist = plt.subplots(1, 1, figsize=(float(img_hist.shape[1])/dpi_hist,float(img_hist.shape[0])/dpi_hist))
# Remove whitespace from around the image
#fig_hist.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax_hist.set_title("Histology viewer")
# Show the histology image  
ax_hist.imshow(img_hist_temp, extent=[0, img_hist.shape[1]*pixdim_hist, img_hist.shape[0]*pixdim_hist, 0])
plt.tick_params(
    axis='both', 
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
ax_hist.format_coord = lambda x, y: ''
plt.show()
# Fix size and location of the figure window
mngr_hist = plt.get_current_fig_manager()
mngr_hist.window.setGeometry(150,300,d2,d1)
        
# User controls 
print('Registration: \n')
print('t: toggle mode where clicks are logged for transform \n')
print('h: toggle overlay of current histology slice \n')
#print('b: toggle to viewing boundaries \n')
print('a: higher quality visualization of boundaries \n')
print('d: delete most recent probe point or transform point \n');

# Lists for the points clicked in atlas and histology
coords_atlas = []
coords_hist = []
# Lists for the points plotted in atlas and histology
redp_atlas = []
redp_hist = []

Edges = np.empty((512,1024,512))
for sl in range(0,1024):
    Edges[:,sl,:] = cv2.Canny(np.uint8((cv_plot[:,sl,:]*255).transpose((1,0,2))),100,200)  

# Reaction to key pressed
def on_key(event):
    if event.key == 't':
        print('Select at least 4 points in the same order in both figures')
        # ATLAS
        # Mouse click function to store coordinates.
        # Leave a red dot when a point is clicked
        # when registering slides        
        def onclick(event):
            global ix, iy
            ix, iy = event.xdata/pixdim, event.ydata/pixdim
            # assign global variable to access outside of function
            global coords_atlas, redp_atlas
            coords_atlas.append((ix, iy))
            redp_atlas.extend(plt.plot(event.xdata, event.ydata, 'ro',markersize=2))
            fig.canvas.draw()
            return
        # Call click func
        fig.canvas.mpl_connect('button_press_event', onclick)  
  
        # HISTOLOGY  
        # Mouse click function to store coordinates.
        # Leave a red dot when a point is clicked
        # when registering slides        
        def onclick_hist(event):
            global xh, yh
            xh, yh = event.xdata/pixdim_hist, event.ydata/pixdim_hist
            # assign global variable to access outside of function
            global coords_hist, redp_hist
            coords_hist.append((xh, yh))
            redp_hist.extend(plt.plot(event.xdata, event.ydata, 'ro',markersize=2))
            fig_hist.canvas.draw()
            return
        # Call click func
        fig_hist.canvas.mpl_connect('button_press_event', onclick_hist)
        
    elif event.key == 'h':
        print('Transform histology to adpat to the atlas')
        
        # get the projective transformation from the set of clicked points
        t = transform.ProjectiveTransform()
        t.estimate(np.float32(coords_atlas),np.float32(coords_hist))
        global img_warped # avoid unbound local error when passed top the next step
        img_warped = transform.warp(img_hist_temp, t, output_shape = (d1,d2), order=1, clip=False)#, mode='constant',cval=float('nan'))
        
# =============================================================================
#         # Show the  transformed figure  
#         fig_trans, ax_trans = plt.subplots(1, 1)#, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl))
#         ax_trans.imshow(img_warped, extent=[0, d1*pixdim, 0, d2*pixdim] )
#         plt.show()
# =============================================================================
    
# =============================================================================
#     elif event.key == 'b':   
# #       SIMPLE OVERLAY
#         fig_g, ax_g = plt.subplots(1, 1) 
#         ax_g.imshow(img_warped, origin="lower", extent=[0, d1*pixdim, d2*pixdim,0])
#         tracker2 = IndexTracker_cg(ax_g, Edges, pixdim, S)
#         fig_g.canvas.mpl_connect('scroll_event', tracker2.onscroll)                
#         ax_g.format_coord = format_coord
#         plt.show()  
#         plt.pause(0.001)
# =============================================================================
        
    elif event.key == 'a':        
        print('Overlay to the atlas')
        # get the edges of the colors defined in the label
        if plane.lower() == 'c':
            edges = cv2.Canny(np.uint8((cv_plot[:,S,:]*255).transpose((1,0,2))),100,200)  
        elif plane.lower() == 's':
            edges = cv2.Canny(np.uint8((cv_plot[S,:,:]*255).transpose((1,0,2))),100,200)
        elif plane.lower() == 'h':    
            edges = cv2.Canny(np.uint8((cv_plot[:,:,S]*255).transpose((1,0,2))),100,200)
            
        fig_grid, ax_grid = plt.subplots(1, 1) 
        CC = np.where(edges == 255)
# =============================================================================
#         GRAY SCALE
#         img = cv2.merge((img_warped,img_warped,img_warped))  # creat RGB image from grayscale
#         img2 = (img).copy()
#         img2[CC] = [0.25,0.41,0.88]  # turn edges to blue
#         ax_grid.imshow(img2, origin="lower")
# =============================================================================
        img2 = (img_warped).copy()
        img2[CC] = 0.5
        ax_grid.imshow(img2, origin="lower", extent=[0, d1*pixdim, 0, d2*pixdim])   
        ax_grid.text(0.15, 0.05, textstr, transform=ax.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
        ax_grid.format_coord = format_coord
        plt.show()
        cursor = mplcursors.cursor(hover=True)
        # Show the names of the regions
        def show_annotation(sel):
            xi, yi = sel.target/pixdim
            if np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),S,int(math.modf(yi)[1])], axis = 1)).size:
                Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),S,int(math.modf(yi)[1])], axis = 1))[0,0]]
            else:
                # display nothing
                Text = ' '
            sel.annotation.set_text(Text)
        cursor.connect('add', show_annotation)   
        mngr_grid = plt.get_current_fig_manager()
        mngr_grid.window.setGeometry(800,300,d2,d1)     
    elif event.key == 'p':        
        print('Register probe track')
        
    elif event.key == 'd':
        print('Delete clicked points')
        coords_atlas.pop(-1) # remove the point from the list
        redp_atlas[-1].remove() # remove the point from the plot
        fig.canvas.draw()
        redp_atlas.pop(-1)
        coords_hist.pop(-1) # remove the point from the list        
        redp_hist[-1].remove() # remove the point from the plot
        fig_hist.canvas.draw()
        redp_hist.pop(-1)

fig.canvas.mpl_connect('key_press_event', on_key)
fig_hist.canvas.mpl_connect('key_press_event', on_key)


















