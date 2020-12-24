# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:33:18 2020

@author: jacopop
"""

from __future__ import print_function

# Import libraries
import os
import os.path
from os import path
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('Qt5Agg')
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import math 
import mplcursors
from nilearn.image import resample_img
from nibabel.affines import apply_affine
from skimage import io, transform
import pickle 

# Functions defined in separate files
# read label file
from Readlabel import readlabel
# from readlabel_customized import readlabel_c
# Allow to navigate in the atlas
from Tracker import IndexTracker, IndexTracker_g, IndexTracker_p
# create objects to svae transformations and probes
from ObjSave import  save_transform, probe_obj, save_probe
        
    
        
# Directory of the processed histology
# for mac user 
processed_histology_folder = '/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed'
# For windows users
# processed_histology_folder = r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\histology\processed'
# for mac user 
# histology = Image.open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/rat_processed.tif').copy()
# For windows users
file_name = str(input('Histology file name: '))
# Mac
img_hist_temp = Image.open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/rat_processed.jpeg').copy()
# Windows
# img_hist_temp = Image.open(os.path.join(processed_histology_folder, file_name+'_processed.jpeg')).copy()
# get the pixel dimension
dpi_hist = img_hist_temp.info['dpi'][1]
pixdim_hist = 25.4/dpi_hist # 1 inch = 25,4 mm
# Windows
# img_hist = cv2.imread(os.path.join(processed_histology_folder, file_name+'_processed.jpeg'),cv2.IMREAD_GRAYSCALE)
# Mac
img_hist = cv2.imread(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/rat_processed.jpeg',cv2.IMREAD_GRAYSCALE)
# Insert the plane of interest
plane = str(input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')).lower()
# Check if the input is correct
while plane != 'c' and plane != 's' and plane != 'h':
    print('Error: Wrong plane name \n')
    plane = str(input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')).lower()

# Paths of the atlas, segmentation and labels
# Atlas
atlas_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v2_pack', 'WHS_SD_rat_T2star_v1.01.nii.gz')
# Mask
mask_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_brainmask_v1.01.nii.gz')
# Segmentation
segmentation_path = os.path.join(r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas', 'WHS_SD_rat_atlas_v4_beta.nii.gz')
# Labels
# Mac
labels_item = open(r"/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.label", "r")
# Windows
# labels_item = open(r"C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v4_beta.label", "r")
labels_index, labels_name, labels_color = readlabel( labels_item )  

# Load the atlas, mask, color and segmentation
# Mac
atlas = nib.load(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v2_pack/WHS_SD_rat_T2star_v1.01.nii.gz')
# Windows
#atlas = nib.load(atlas_path)
atlas_header = atlas.header
# get pixel dimension
pixdim = atlas_header.get('pixdim')[1]
#atlas_data = atlas.get_fdata()
#atlas_affine = atlas.affine
# Windows
# atlas_data = np.load('atlas_data_masked.npy')
# mac
atlas_data = np.load('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Rat/RatBrain/atlas_data_masked.npy')
# Mac
#mask = nib.load(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_brainmask_v1.01.nii.gz')
# Windows
#mask = nib.load(mask_path)
#mask_data = mask.get_fdata()[:,:,:,0]
# Mac
segmentation = nib.load('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.nii.gz')
# Windows
#segmentation = nib.load(segmentation_path)
segmentation_data = segmentation.get_fdata()

# Atlas in RGB colors according with the label file
# cv_plot = np.load('cv_plot.npy')/255
# mac
cv_plot = np.load('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Rat/RatBrain/cv_plot.npy')/255
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

# Display the ATLAS
# resolution
dpi_atl = 25.4/pixdim
# Bregma coordinates
textstr = 'Bregma (mm): c = %.3f, h = %.3f, s = %.3f \nBregma (voxels): c = 653, h = 440, s = 246' %( 653*pixdim, 440*pixdim, 246*pixdim)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# Figure
fig, ax = plt.subplots(1, 1)#, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl), dpi=dpi_atl)
# scroll cursor
tracker = IndexTracker(ax, atlas_data, pixdim, plane)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# place a text box with bregma coordinates in bottom left in axes coords
ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
if plane == 'c':
    # dimensions
    d1 = 512
    d2 = 512
    d3 = 1024
    # display the coordinates relative to the bregma when hovering with the cursor
    def format_coord(x, y):
        AP = tracker.ind*pixdim - 653*pixdim
        ML = y - 246*pixdim
        Z = x - 440*pixdim
        if ML >0:        
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, ML, Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, ML, Z)    
    ax.format_coord = format_coord
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
elif plane == 's':
    # dimensions
    d1 = 1024 
    d2 = 512    
    d3 = 512
    # display the coordinates relative to the bregma when hovering with the cursor
    def format_coord(x, y):
        AP = y - 653*pixdim
        ML = tracker.ind*pixdim - 246*pixdim
        Z = x - 440*pixdim
        if ML >0:        
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, ML, Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, ML, Z)    
    ax.format_coord = format_coord
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
elif plane == 'h':
    # dimensions
    d1 = 512
    d2 = 1024
    d3 = 512    
    # display the coordinates relative to the bregma when hovering with the cursor
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

# Show the HISTOLOGY
# Set up figure
fig_hist, ax_hist = plt.subplots(1, 1, figsize=(float(img_hist.shape[1])/dpi_hist,float(img_hist.shape[0])/dpi_hist))
ax_hist.set_title("Histology viewer")
# Show the histology image  
ax_hist.imshow(img_hist_temp, extent=[0, img_hist.shape[1]*pixdim_hist, img_hist.shape[0]*pixdim_hist, 0])
# Remove axes tick
plt.tick_params(
    axis='both', 
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
# Remove cursor position
ax_hist.format_coord = lambda x, y: ''
plt.show()
# Fix size and location of the figure window
mngr_hist = plt.get_current_fig_manager()
mngr_hist.window.setGeometry(150,300,d2,d1)
        
# User controls 
print('\n Registration: \n')
print('t: toggle mode where clicks are logged for transform \n')
print('h: toggle overlay of current histology slice \n')
print('x: save transform and current atlas location')
image_name = str(input('Enter transformed image name: '))
print('\nr: toggle mode where clicks are logged for probe or switch probes \n')
print('n: add a new probe \n')
print('e: save current probe \n')
print('p: switch probe \n')
print('w: enable/disable probe viewer mode for current probe  \n')
# =============================================================================
# print('l: load transform for current slice; press again to load probe points \n');
# ============================================================================
print('\n Viewing modes: \n')
print('a: higher quality visualization of boundaries \n')
print('b: toggle to viewing boundaries \n')
print('d: delete most recent transform point \n')
print('c: delete most recent probe point \n')
print('g: toggle gridlines \n')
print('v: toggle to color atlas mode \n')

# Lists for the points clicked in atlas and histology
coords_atlas = []
coords_hist = []
coords_probe_temp_w = []
coords_probe_temp_g = []
coords_probe_temp_p = []          
coords_probe_temp_b = []
coords_probe_temp_y = []
coords_probe_temp_o = []
coords_probe_temp_r = []
# Object for clicked probes
coords_probe = probe_obj()
# Lists for the points plotted in atlas and histology
redp_atlas = []
redp_hist = []
# List of probe points
p_probe_trans = []
p_probe_grid = []
# Initialize probe counter and selecter
probe_counter = 0
probe_selecter = 0

# get the edges of the colors defined in the label
# Windows
# Edges = np.load('Edges.npy')
# mac
Edges = np.load('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Rat/RatBrain/Edges.npy')
# =============================================================================
# Edges = np.empty((512,1024,512))
# for sl in range(0,1024):
#     Edges[:,sl,:] = cv2.Canny(np.uint8((cv_plot[:,sl,:]*255).transpose((1,0,2))),100,200)  
# 
# =============================================================================
# Set up the figure    
plt.ioff()
fig_trans, ax_trans = plt.subplots(1, 1)
#fig_grid, ax_grid = plt.subplots(1, 1)
# plt.ioff()
# Reaction to key pressed
def on_key(event):
    if event.key == 't': 
        print('Select at least 4 points in the same order in both figures')
        # ATLAS
        # Mouse click function to store coordinates. Leave a red dot when a point is clicked
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
        # Mouse click function to store coordinates. Leave a red dot when a point is clicked      
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
        global img_warped, fig_trans, ax_trans # avoid unbound local error when passed top the next step
        img_warped = transform.warp(img_hist_temp, t, output_shape = (d1,d2), order=1, clip=False)#, mode='constant',cval=float('nan'))
        # Show the  transformed figure  
        #fig_trans, ax_trans = plt.subplots(1, 1)#, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl))
        ax_trans.imshow(img_warped, origin="lower", extent=[0, d1*pixdim, 0, d2*pixdim] )
        ax_trans.set_title("Histology adapted to atlas")
        plt.show()
    
    elif event.key == 'b':
        print('Simple overlay')
#       SIMPLE OVERLAY
        global tracker2, fig_g
        fig_g, ax_g = plt.subplots(1, 1) 
        ax_g.imshow(img_warped, origin="lower", extent=[0, d1*pixdim, d2*pixdim,0])
        tracker2 = IndexTracker_g(ax_g, Edges, pixdim, plane, tracker.ind)
        fig_g.canvas.mpl_connect('scroll_event', tracker2.onscroll)  
        #ax_g.format_coord = format_coord
        ax_g.set_title("Histology and atlas overlayed")
        plt.show()  
        # Remove axes tick
        plt.tick_params(axis='both', which='both', bottom=False, left=False, top=False, labelbottom=False, labelleft=False) 
          
    elif event.key == 'a':        
        print('Overlay to the atlas')
        # get the edges of the colors defined in the label
        if plane == 'c':
            edges = cv2.Canny(np.uint8((cv_plot[:,tracker.ind,:]*255).transpose((1,0,2))),100,200)  
        elif plane == 's':
            edges = cv2.Canny(np.uint8((cv_plot[tracker.ind,:,:]*255).transpose((1,0,2))),100,200)
        elif plane == 'h':    
            edges = cv2.Canny(np.uint8((cv_plot[:,:,tracker.ind]*255).transpose((1,0,2))),100,200)
        global img2, fig_grid, ax_grid
        fig_grid, ax_grid = plt.subplots(1, 1)
        # position of the lines
        CC = np.where(edges == 255)
        img2 = (img_warped).copy()
        # get the lines in the warped figure
        img2[CC] = 0.5
        overlay = ax_grid.imshow(img2, origin="lower", extent=[0, d1*pixdim, 0, d2*pixdim])   
        ax_grid.text(0.15, 0.05, textstr, transform=ax.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
        ax_grid.format_coord = format_coord
        ax_grid.set_title("Histology and atlas overlayed")
        plt.show()
        cursor = mplcursors.cursor(overlay, hover=True)
        # Show the names of the regions
        def show_annotation(sel):
            xi, yi = sel.target/pixdim
            if np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1)).size:
                Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1))[0,0]]
            else:
                # display nothing
                Text = ' '
            sel.annotation.set_text(Text)
        cursor.connect('add', show_annotation)   
        mngr_grid = plt.get_current_fig_manager()
        mngr_grid.window.setGeometry(800,300,d2,d1)   
        
    elif event.key == 'd':
        print('Delete clicked point')
        coords_atlas.pop(-1) # remove the point from the list
        redp_atlas[-1].remove() # remove the point from the plot
        fig.canvas.draw()
        redp_atlas.pop(-1)
        coords_hist.pop(-1) # remove the point from the list        
        redp_hist[-1].remove() # remove the point from the plot
        fig_hist.canvas.draw()
        redp_hist.pop(-1)
        
    elif event.key == 'x':
        print('Save image and slice')            
        # The Transformed images will be saved in a subfolder of process histology called transformations
        path_transformed = os.path.join(processed_histology_folder, 'transformations')
        if not path.exists(os.path.join(processed_histology_folder, 'transformations')):
            os.mkdir(path_transformed)
        # Create and save slice, clicked points, and image info                                 
        S = save_transform(tracker.ind, [coords_hist, coords_atlas], img2, img_warped)        # Saving the object
        with open(os.path.join(path_transformed, image_name+'.pkl'), 'wb') as f: 
            pickle.dump(S, f)
        # Save the images
        fig_trans.savefig(os.path.join(path_transformed, image_name+'_Transformed_withoutlines.jpeg'))
                        
    elif event.key == 'v':
        print('Colored Atlas on')
        global fig_color
        fig_color, ax_color = plt.subplots(1, 1) 
        ax_color.imshow(img2, extent=[0, d1*pixdim, 0, d2*pixdim])
        if plane == 'c':
            ax_color.imshow(cv_plot[:,tracker.ind,:].transpose((1,0,2)), origin="lower", extent=[0, d1*pixdim, d2*pixdim, 0], alpha = 0.5)                        
        elif plane == 's':
            ax_color.imshow(cv_plot[tracker.ind,:,:].transpose((1,0,2)), origin="lower", extent=[0, d1*pixdim, d2*pixdim, 0], alpha = 0.5)
        elif plane == 'h':
            ax_color.imshow(cv_plot[:,:,tracker.ind].transpose((1,0,2)), origin="lower", extent=[0, d1*pixdim, d2*pixdim, 0], alpha = 0.5)     
        ax_color.set_title("Histology and colored atlas")
        plt.show()
        
    elif event.key == 'r':     
        print('Register probe 1 (green)')
        try:
            plt.close(fig_g)
        except:
            pass
        try: 
            plt.close(fig_color)
        except:
            pass
        # probes have different colors 
        global probe_colors                            
        probe_colors = ['green', 'purple', 'blue', 'yellow', 'orange', 'red']
        # plot  point and register all the clicked points
        def onclick_probe(event):
            global px, py
            px, py = event.xdata/pixdim, event.ydata/pixdim
            # assign global variable to access outside of function
            global coords_probe_temp_w, coords_probe_temp_g, coords_probe_temp_p, coords_probe_temp_b, coords_probe_temp_y, coords_probe_temp_o, coords_probe_temp_r,  p_probe_grid, p_probe_trans
            if probe_counter == 0:
                coords_probe_temp_w.append((px, py)) 
                p_probe_grid.extend(ax_grid.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                p_probe_trans.extend(ax_trans.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_w)
            elif probe_counter == 1:
                coords_probe_temp_g.append((px, py))
                p_probe_grid.extend(ax_grid.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                p_probe_trans.extend(ax_trans.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_g)
            elif probe_counter == 2:
                coords_probe_temp_p.append((px, py))    
                p_probe_grid.extend(ax_grid.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                p_probe_trans.extend(ax_trans.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_p)
            elif probe_counter == 3:
                coords_probe_temp_b.append((px, py))
                p_probe_grid.extend(ax_grid.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                p_probe_trans.extend(ax_trans.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_b)
            elif probe_counter == 4:
                coords_probe_temp_y.append((px, py))
                p_probe_grid.extend(ax_grid.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                p_probe_trans.extend(ax_trans.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_y)
            elif probe_counter == 5:
                coords_probe_temp_o.append((px, py))
                p_probe_grid.extend(ax_grid.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                p_probe_trans.extend(ax_trans.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_o)
            fig_grid.canvas.draw()
            fig_trans.canvas.draw()
            return
        # Call click func
        fig_trans.canvas.mpl_connect('button_press_event', onclick_probe) 
        
        def on_key2(event):            
            if event.key == 'n':
                # add a new probe, the function in defined in onclick_probe
                global probe_counter
                if probe_counter+1 <=len(probe_colors):
                    probe_counter +=  1                                                                                                 
                    print('probe %d added (%s)' %(probe_counter, probe_colors[probe_counter]))
                else:
                    print('Cannot add more probes')
                    probe_counter = len(probe_colors)
                    
            elif event.key == 'c':
                print('Delete clicked probe point')
                if len(getattr(coords_probe,probe_colors[0]))!= 0:
                    if len(getattr(coords_probe,probe_colors[probe_counter])) != 0:
                        getattr(coords_probe,probe_colors[probe_counter]).pop(-1) # remove the point from the list
                        p_probe_trans[-1].remove() # remove the point from the plot
                        fig_trans.canvas.draw()
                        p_probe_trans.pop(-1)                        
                        p_probe_grid[-1].remove() # remove the point from the plot
                        fig_grid.canvas.draw()           
                        p_probe_grid.pop(-1)
                    elif len(getattr(coords_probe,probe_colors[probe_counter])) == 0:
                        probe_counter -=1
                        try:
                            getattr(coords_probe,probe_colors[probe_counter]).pop(-1) # remove the point from the list
                            p_probe_trans[-1].remove() # remove the point from the plot
                            fig_trans.canvas.draw()                                        
                            p_probe_trans.pop(-1)
                            p_probe_grid[-1].remove() # remove the point from the plot
                            fig_grid.canvas.draw()                        
                            p_probe_grid.pop(-1)
                        except:
                            pass
# =============================================================================
#             elif event.key == 'p':
#                 print( 'Change probe' )
#                 global probe_counter
#                 if probe_counter-1 > 0:
#                     probe_counter -=  1                                                                                                 
#                     print('probe %d selected (%s)' %(probe_counter+1, probe_colors[probe_counter]))
#                 elif probe_counter == 0:
#                     probe_counter +=1 
#                     print('probe %d selected (%s)' %(probe_counter+1, probe_colors[probe_counter]))
# 
# 
# =============================================================================
                        
        fig_trans.canvas.mpl_connect('key_press_event', on_key2)
        
    elif event.key == 'e':
        print('Probe points saved')        
        path_probes = os.path.join(processed_histology_folder, 'probes')
        if not path.exists(os.path.join(processed_histology_folder, 'probes')):
            os.mkdir(path_probes)
        # Create and save slice, clicked probes
        P = save_probe(tracker.ind, coords_probe, plane, probe_counter)        # Saving the object
# =============================================================================
#         with open(os.path.join(path_probes, image_name+'probes.pkl'), 'wb') as F: 
#             pickle.dump(P, F)# Create and save slice, clicked points, and image info    
# =============================================================================  
        # MAC    
        with open('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes/1probes.pkl', 'wb') as F: 
            pickle.dump(P, F)# Create and save slice, clicked points, and image info 
            
    elif event.key == 'w':
        try:   
            global probe_selecter
            print('probe %d view mode' %(probe_selecter+1))
            L = getattr(coords_probe,probe_colors[probe_selecter])
            probe_x = []
            probe_y = []
            for i in range(len(L)):
                probe_x.append(L[i][0]*pixdim)
                probe_y.append(L[i][1]*pixdim)
            m, b = np.polyfit(probe_x, probe_y, 1)
            fig_probe, ax_probe = plt.subplots(1, 1)  
            trackerp = IndexTracker_p(ax_probe, atlas_data, pixdim, plane, tracker.ind)
            fig_probe.canvas.mpl_connect('scroll_event', trackerp.onscroll)        
            ax_probe.text(0.15, 0.05, textstr, transform=ax_probe.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
            ax_probe.format_coord = format_coord
            ax_probe.set_title("Probe viewer")
            plt.show()
            cursor = mplcursors.cursor(fig_probe, hover=True)
            # Show the names of the regions
            def show_annotation(sel):
                xi, yi = sel.target/pixdim
                if np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1)).size:
                    Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1))[0,0]]
                else:
                    # display nothing
                    Text = ' '
                sel.annotation.set_text(Text)
            cursor.connect('add', show_annotation)   
            mngr_probe = plt.get_current_fig_manager()
            mngr_probe.window.setGeometry(800,300,d2,d1)    
            # plot the clicked points
            plt.scatter(probe_x, probe_y, color=probe_colors[probe_selecter], s=2)#, marker='o', markersize=1)
            # plot the probe
            plt.plot(np.array(probe_x), m*np.array(probe_x) + b,color=probe_colors[probe_selecter], linestyle='dashed', linewidth=0.8)
            probe_selecter +=1        
        except:
            print('No more probes to visualize')
            pass

        
            
fig.canvas.mpl_connect('key_press_event', on_key)
fig_hist.canvas.mpl_connect('key_press_event', on_key)
#fig_grid.canvas.mpl_connect('key_press_event', on_key)
fig_trans.canvas.mpl_connect('key_press_event', on_key)
