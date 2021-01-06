# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:20:33 2020

@author: jacopop
"""


from __future__ import print_function

# Import libraries
import math 
import os
import os.path
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import pickle 
from skimage import measure
from collections import OrderedDict, Counter
from tabulate import tabulate
import mplcursors
    
# 3d Brain
from vedo import Volume as VedoVolume
import vedo 
from vedo import buildLUT, Sphere, show, settings
from vedo.mesh import Mesh, merge
from vedo import *

# fit the probe
from skspatial.objects import Line
#from skspatial.plotting import plot_3d

from ObjSave import  probe_obj, save_probe
from Tracker import IndexTracker, IndexTracker_g, IndexTracker_p, IndexTracker_b, IndexTracker_c
# read label file
from Readlabel import readlabel


print('Controls: \n')
print('--------- \n')
print('scroll: move between slices \n')
print('g: add/remove gridlines \n')
print('b: add name of current region extent \n')
print('a: toggle viewing boundaries \n')
print('v: toggle color atlas mode \n')
print('p: enable/disable mode where clicks are logged for probe or switch probes \n')
print('n: trace a new probe \n')
print('l: load saved probe points \n')
print('s: save current probe \n')
print('w: enable/disable probe viewer mode for current probe  \n')
print('d: delete most recent probe point \n')
print('up: scroll through A/P angles \n')
print('right: scroll through M/L angles \n')


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
labels_index, labels_name, labels_color, labels_initials = readlabel( labels_item )  

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
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, abs(ML), Z)    
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
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, abs(ML), Z)    
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
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, abs(ML), Z)    
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
mngr.window.setGeometry(600,200,d2*2,d1*2)      
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


# Lists for the points clicked in atlas and histology
coords_atlas = []
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
p_probe= []
# Initialize probe counter and selecter
probe_counter = 0
probe_selecter = 0


def on_key(event):
    if event.key == 'a':
        print('View boundaries')
        global tracker2
        tracker2 = IndexTracker_b(ax, Edges, pixdim, plane, tracker.ind)    
        fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
        plt.show() 
    elif event.key == 'v':
        print('View colors')
        global tracker3
        tracker3 = IndexTracker_c(ax, cv_plot, pixdim, plane, tracker.ind)        
        fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)  
        #ax_g.format_coord = format_coord 
        plt.show()
    elif event.key == 'b':   
        print("Show region's name")
        # Show the names of the regions
        global cursor
        cursor = mplcursors.cursor(hover=True)
        def show_annotation(sel):
            xi, yi = sel.target/pixdim
            if plane == 'c':
                if np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1)).size:
                    Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1))[0,0]]
                else:
                    # display nothing
                    Text = ' '                
            elif plane == 's':
                if np.argwhere(np.all(labels_index == segmentation_data[tracker.ind,int(math.modf(xi)[1]),int(math.modf(yi)[1])], axis = 1)).size:
                    Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[tracker.ind,int(math.modf(xi)[1]),int(math.modf(yi)[1])], axis = 1))[0,0]]
                else:
                    # display nothing
                    Text = ' '
            elif plane == 'h':
                if np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),int(math.modf(yi)[1]),tracker.ind], axis = 1)).size:
                    Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),int(math.modf(yi)[1]),tracker.ind], axis = 1))[0,0]]
                else:
                    # display nothing
                    Text = ' '                    
            sel.annotation.set_text(Text)
        cursor.connect('add', show_annotation) 

    elif event.key == 'r':     
        print('Register probe 1 (green)')
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
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_w)
            elif probe_counter == 1:
                coords_probe_temp_g.append((px, py))
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_g)
            elif probe_counter == 2:
                coords_probe_temp_p.append((px, py))    
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_p)
            elif probe_counter == 3:
                coords_probe_temp_b.append((px, py))
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_b)
            elif probe_counter == 4:
                coords_probe_temp_y.append((px, py))
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_y)
            elif probe_counter == 5:
                coords_probe_temp_o.append((px, py))
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_o)
            fig.canvas.draw()
            return
        # Call click func
        fig.canvas.mpl_connect('button_press_event', onclick_probe) 
        
# =============================================================================
#         def on_key2(event):            
#             if event.key == 'n':
#                 # add a new probe, the function in defined in onclick_probe
#                 global probe_counter
#                 if probe_counter+1 <=len(probe_colors):
#                     probe_counter +=  1                                                                                                 
#                     print('probe %d added (%s)' %(probe_counter, probe_colors[probe_counter]))
#                 else:
#                     print('Cannot add more probes')
#                     probe_counter = len(probe_colors)
#                     
#             elif event.key == 'c':
#                 print('Delete clicked probe point')
#                 if len(getattr(coords_probe,probe_colors[0]))!= 0:
#                     if len(getattr(coords_probe,probe_colors[probe_counter])) != 0:
#                         getattr(coords_probe,probe_colors[probe_counter]).pop(-1) # remove the point from the list
#                         p_probe[-1].remove() # remove the point from the plot
#                         fig.canvas.draw()
#                         p_probe.pop(-1)                        
#                     elif len(getattr(coords_probe,probe_colors[probe_counter])) == 0:
#                         probe_counter -=1
#                         try:
#                             getattr(coords_probe,probe_colors[probe_counter]).pop(-1) # remove the point from the list
#                             p_probe[-1].remove() # remove the point from the plot
#                             fig.canvas.draw()                                        
#                             p_probe.pop(-1)
# 
#                         except:
#                             pass
# =============================================================================
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
                        
# =============================================================================
#         fig_trans.canvas.mpl_connect('key_press_event', on_key2)
#         
#     elif event.key == 'e':
#         print('Probe points saved')        
#         path_probes = os.path.join(processed_histology_folder, 'probes')
#         if not path.exists(os.path.join(processed_histology_folder, 'probes')):
#             os.mkdir(path_probes)
#         # Create and save slice, clicked probes
#         P = save_probe(tracker.ind, coords_probe, plane, probe_counter)        # Saving the object
# # =============================================================================
# #         with open(os.path.join(path_probes, image_name+'probes.pkl'), 'wb') as F: 
# #             pickle.dump(P, F)# Create and save slice, clicked points, and image info    
# # =============================================================================  
#         # MAC    
#         with open('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes/1probes.pkl', 'wb') as F: 
#             pickle.dump(P, F)# Create and save slice, clicked points, and image info 
#             
#     elif event.key == 'w':
#         try:   
#             global probe_selecter
#             print('probe %d view mode' %(probe_selecter+1))
#             L = getattr(coords_probe,probe_colors[probe_selecter])
#             probe_x = []
#             probe_y = []
#             for i in range(len(L)):
#                 probe_x.append(L[i][0]*pixdim)
#                 probe_y.append(L[i][1]*pixdim)
#             m, b = np.polyfit(probe_x, probe_y, 1)
#             fig_probe, ax_probe = plt.subplots(1, 1)  
#             trackerp = IndexTracker_p(ax_probe, atlas_data, pixdim, plane, tracker.ind)
#             fig_probe.canvas.mpl_connect('scroll_event', trackerp.onscroll)        
#             ax_probe.text(0.15, 0.05, textstr, transform=ax_probe.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
#             ax_probe.format_coord = format_coord
#             ax_probe.set_title("Probe viewer")
#             plt.show()
#             cursor = mplcursors.cursor(fig_probe, hover=True)
#             # Show the names of the regions
#             def show_annotation(sel):
#                 xi, yi = sel.target/pixdim
#                 if np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1)).size:
#                     Text = labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(xi)[1]),tracker.ind,int(math.modf(yi)[1])], axis = 1))[0,0]]
#                 else:
#                     # display nothing
#                     Text = ' '
#                 sel.annotation.set_text(Text)
#             cursor.connect('add', show_annotation)   
#             mngr_probe = plt.get_current_fig_manager()
#             mngr_probe.window.setGeometry(800,300,d2,d1)    
#             # plot the clicked points
#             plt.scatter(probe_x, probe_y, color=probe_colors[probe_selecter], s=2)#, marker='o', markersize=1)
#             # plot the probe
#             print(probe_x)
#             plt.plot(np.array(sorted(probe_x)), m*np.array(sorted(probe_x)) + b,color=probe_colors[probe_selecter], linestyle='dashed', linewidth=0.8)
#             probe_selecter +=1        
#         except:
#             print('No more probes to visualize')
#             pass
# =============================================================================

        
            
fig.canvas.mpl_connect('key_press_event', on_key)



        



# =============================================================================
# 
# 
# # Probe colors
# probe_colors = ['green', 'purple', 'blue', 'yellow', 'orange', 'red']
# 
# # Windows
# # =============================================================================
# # processed_histology_folder = r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\histology\processed'
# # path_probes = os.path.join(processed_histology_folder, 'probes')
# # path_transformed = os.path.join(processed_histology_folder, 'transformations')
# # =============================================================================
# 
# # Mac
# processed_histology_folder = r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed'
# path_probes = r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes'
# path_transformed = '/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/transformations'
# 
# # get the all the files in the probe folder
# files_probe = os.listdir(path_probes)
# files_transformed = os.listdir(path_transformed)
# 
# L = probe_obj()
# LINE_FIT = probe_obj() 
# pr = probe_obj()
# xyz = probe_obj()
# P = []
# color_used_t = []
# # =============================================================================
# # for f in files_probe:
# #     # WINDOWS
# #     P.append(pickle.load(open(os.path.join(path_probes, f), "rb")))
# # =============================================================================
#     # MAC
# P.append(pickle.load(open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes/1probes.pkl', "rb")))
# P.append(pickle.load(open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes/2probes.pkl', "rb")))
# # LL = pickle.load(open(os.path.join(path_probes, '1probes.pkl'), "rb"))    
# probe_counter = P[0].Counter
# 
# # If I have several probes
# for j in range(len(probe_colors)):    
#     # get the probe coordinates and the region's names
#     probe_x = []
#     probe_y = []
#     probe_z = []
#     for k in range(len(P)):
#         try:
#             PC = getattr(P[k].Probe, probe_colors[j])
#             if P[k].Plane == 'c':
#                 for i in range(len(PC)):
#                     probe_x.append(PC[i][0])
#                     probe_y.append(P[k].Slice)
#                     probe_z.append(PC[i][1])
#             elif P[k].Plane == 'c':
#                 for i in range(len(PC)):
#                     probe_x.append(P[k].Slice)
#                     probe_y.append(PC[i][0])
#                     probe_z.append(PC[i][1])  
#             elif P[k].Plane == 'c':
#                 for i in range(len(PC)):        
#                     probe_x.append(PC[i][0])
#                     probe_y.append(PC[i][1])        
#                     probe_z.append(P[k].Slice)
#             pts = np.array((probe_x, probe_y, probe_z)).T
# 
#             # fit the probe
#             line_fit = Line.best_fit(pts)
#             # line equations, to derive the starting and end point of the line (aka probe)
#             z1 = max(pts[:,2])
#             x1 = line_fit.point[0]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
#             y1 = line_fit.point[1]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
#             z2 = min(pts[:,2])
#             x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
#             y2= line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
#             # get the line to plot
#             l = vedo.Line([x1, y1, z1],[x2, y2, z2],c = probe_colors[j], lw = 2)
#             # clicked points to display
#             pp = vedo.Points(pts, c = probe_colors[j]) #fast    
#             setattr(xyz,probe_colors[j], [[x1, y1, z1], [x2, y2, z2]])
#             setattr(pr, probe_colors[j], pp)
#             setattr(L, probe_colors[j], l)
#             setattr(LINE_FIT, probe_colors[j], line_fit)
#             color_used_t.append(probe_colors[j])
#         except:
#             pass  
# 
# # get only the unique color in order                  
# color_used = list(OrderedDict.fromkeys(color_used_t))
# n = len(color_used)
# 
# # load the brain regions
# Edges = np.load('Edges.npy')
# edges = Edges.T
# coords = np.array(np.where(edges == 255))
# # Manage Points cloud
# points = vedo.pointcloud.Points(coords)
# # Create the mesh
# mesh = Mesh(points)
# 
# # create some dummy data array to be associated to points
# data = mesh.points()[:,2]  # pick z-coords, use them as scalar data
# # build a custom LookUp Table of colors:
# lut = buildLUT([
#                 (512, 'lightgrey', 0.01 ),
#                ],
#                vmin=0, belowColor='lightblue',
#                vmax= 512, aboveColor='grey',
#                nanColor='red',
#                interpolate=False,
#               )
# mesh.cmap(lut, data)
# 
# dist = []
# 
# # To plot the probe with colors
# 
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, aspect='equal')
# # compute and display the insertion angle for each probe
# for i in range(0,n):
#     line_fit = getattr(LINE_FIT, color_used[i])
#     deg_lat = math.degrees(math.atan(line_fit.direction[0]))
#     deg_ant = math.degrees(math.atan(line_fit.direction[1]))
#     print('\n\nAnalyze %s probe: \n ' %color_used[i])
#     print('Estimated %s probe insertion angle: ' %color_used[i])
#     print('%.2f degrees in the anterior direction' %deg_ant)
#     print('%.2f degrees in the lateral direction\n' %deg_lat)
# 
#     # Get the brain regions traversed by the probe
#     X1 = getattr(xyz, color_used[i])[0]
#     X2 = getattr(xyz, color_used[i])[1]
#     s = min([int(math.modf(X1[2])[1]),int(math.modf(X2[2])[1])]) # starting point
#     f = max([int(math.modf(X1[2])[1]),int(math.modf(X2[2])[1])]) # ending point
#     # get lenght of the probe
#     dist.append(np.linalg.norm(f-s))
#     regions = []
#     colori = []
#     initials = []
#     index = []
#     channels = []
#     for z in range(s,f):
#         x = line_fit.point[0]+((z-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
#         y = line_fit.point[1]+((z-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
#         regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
#         colori.append(labels_color[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
#         initials.append(labels_initial[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
#         index.append(segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])])
#         #channels.append(0)
#         # count the number of elements in each region to 
#     counter_regions = dict(Counter(regions))    
#     regioni = list(OrderedDict.fromkeys(regions))
#     iniziali = list(OrderedDict.fromkeys(initials))
#     indici = list(OrderedDict.fromkeys(index))
#     
#     LL = [regioni,  iniziali]
#     headers = [' Regions traversed', 'Initials']
#     numpy_array = np.array(LL)
#     transpose = numpy_array.T
#     transpose_list = transpose.tolist()
#     print(tabulate(transpose_list, headers, floatfmt=".2f"))
#     cc = 0
#     jj = 0
#     for re in regioni:
#         #print(re)
#         # proportion of the probe in the given region
#         dist_prop = counter_regions[re]/dist[i]
#         color_prop = labels_color[np.argwhere(np.array(labels_name)== re)]
#         # plot the probe with the colors of the region traversed
#         ax1.add_patch(patches.Rectangle((70*i+20, cc), 20, dist_prop*dist[i], color=color_prop[0][0]/255))
#         plt.text(70*i+20, max(dist)+2, 'Probe %d (%s)'%(i+1, color_used[i]), fontsize=7.5)
#         plt.text(70*i+45, cc+round(dist_prop*dist[i]/2), '%s %d'%(iniziali[jj], indici[jj]), fontsize=5.5)        
#         jj +=1
#         cc = dist_prop*dist[i] + cc    
# lims = (0,max(dist))
# plt.ylim(lims)
# plt.xlim((0,70*n+20))
# plt.axis('off')
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # plot all the probes together
# if n==1:
#      show(mesh, getattr(pr,color_used[0]), getattr(L, color_used[0]), __doc__,
#      axes=0, viewup="z", bg='white',
#      )        
# elif n==2:     
#      show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(L, color_used[0]), getattr(L, color_used[1]), __doc__,
#      axes=0, viewup="z", bg='white',
#      )
# elif n == 3:
#      show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(L, color_used[0]),getattr(L, color_used[1]), getattr(L, color_used[2]), __doc__,
#      axes=0, viewup="z",
#      bg='white', 
#      )
# elif n == 4:
#      show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), __doc__,
#      axes=0, viewup="z", bg='white',
#      )
# elif n==5:
#      show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), getattr(L, color_used[4]),  __doc__,
#      axes=0, viewup="z", bg='white',
#      )
# elif n == 6:
#      show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(pr,color_used[5]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), getattr(L, color_used[4]), getattr(L, color_used[5]), __doc__,
#      axes=0, viewup="z", bg='white',
#      )
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# =============================================================================
