from __future__ import print_function

# Import libraries
import math 
import os
import os.path
from os import path
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
from scipy.spatial import distance
    
# 3d Brain
from vedo import Volume as VedoVolume
import vedo 
from vedo import buildLUT, Sphere, show, settings
from vedo.mesh import Mesh, merge
from vedo import *

# fit the probe
from skspatial.objects import Line
#from skspatial.plotting import plot_3d

from ObjSave import  probe_obj, save_probe_insertion
from Tracker import IndexTracker, IndexTracker_g, IndexTracker_pi, IndexTracker_b, IndexTracker_c
# read label file
from Readlabel import readlabel

# PROBE
max_probe_length = 10 # maximum length of probe shank is 10mm
probe_widht = 0.07  
probe_thickness = 0.024
probe_tip_length = 0.175  
total_electrodes = 960 # total number of recording sites
electrode = 0.012 # Electrode size is 12x12 micron
vert_el_dist = 0.02 
# There are 2 electrodes every 0.02 mm


print('\nControls: \n')
print('--------------------------- \n')
print('scroll: move between slices \n')
print('g: add/remove gridlines \n')
print('b: add/remove name of current region \n')
print('a: add/remove viewing boundaries \n')
print('v: add/remove atlas color \n')
print('r: toggle mode where clicks are logged for probe \n')
print('n: trace a new probe \n')
print('u: load saved probe points \n')   ##
print('e: save probes \n') 
print('w: enable/disable probe viewer mode for current probe  \n') 
print('c: delete most recent probe point \n')
# =============================================================================
# print('up: scroll through A/P angles \n') ##
# print('right: scroll through M/L angles \n') ##
# =============================================================================
print('--------------------------- \n')

flag = 0

# Directory of the processed histology
path_probe_insertion = '/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Probe_Insertion'
if not path.exists(path_probe_insertion):
    os.mkdir(path_probe_insertion)

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
    # display the coordinates relative to the bregma when hovering with the cursor
    def format_coord(x, y):
        AP = tracker.ind*pixdim - 653*pixdim
        ML = x - 246*pixdim
        Z = y - 440*pixdim
        if ML >0:        
            return 'AP=%1.2f, ML=R%1.2f, z=%1.2f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.2f, ML=L%1.2f, z=%1.2f'%(AP, abs(ML), Z)    
    ax.format_coord = format_coord
elif plane == 's':
    # dimensions
    d2 = 1024 
    d1 = 512    
    # display the coordinates relative to the bregma when hovering with the cursor
    def format_coord(x, y):
        AP = x - 653*pixdim
        ML = tracker.ind*pixdim - 246*pixdim
        Z = y - 440*pixdim
        if ML >0:        
            return 'AP=%1.2f, ML=R%1.2f, z=%1.2f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.2f, ML=L%1.2f, z=%1.2f'%(AP, abs(ML), Z)    
    ax.format_coord = format_coord
elif plane == 'h':
    # dimensions
    d2 = 512
    d1 = 1024
    # display the coordinates relative to the bregma when hovering with the cursor
    def format_coord(x, y):
        AP = y - 653*pixdim
        ML = x - 246*pixdim        
        Z = tracker.ind*pixdim - 440*pixdim
        if ML >0:        
            return 'AP=%1.2f, ML=R%1.2f, z=%1.2f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.2f, ML=L%1.2f, z=%1.2f'%(AP, abs(ML), Z)    
    ax.format_coord = format_coord
plt.show()    

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
# List of probe points
p_probe= []
# Initialize probe counter and selecter
probe_counter = 0
probe_selecter = 0
probe_selecter_u = 0

Pp = []

flag_color = 0 
flag_boundaries = 0
flag_names = 0

def on_key(event):   
    global flag_boundaries,flag_color, flag_names
    if event.key == 'a':
        global  tracker, ax
        if flag_boundaries == 0:
            print('View boundaries on')
            global tracker2 
            tracker2 = IndexTracker_b(ax, Edges, pixdim, plane, tracker.ind)    
            fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
            plt.show() 
            flag_boundaries =  1
        elif flag_boundaries == 1:
            print('View boundaries off')
            fig.delaxes(ax)
            ax.clear()
            plt.draw()
            fig.add_axes(ax)
            plt.draw()
            tracker = IndexTracker(ax, atlas_data, pixdim, plane)
            fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
            plt.show()  
            flag_boundaries = 0            
    elif event.key == 'v':
        if flag_color == 0:
            print('View colors on')
            global tracker3
            tracker3 = IndexTracker_c(ax, cv_plot, pixdim, plane, tracker.ind)        
            fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)  
            #ax_g.format_coord = format_coord 
            plt.show()
            flag_color = 1
        elif flag_color == 1:
            print('View colors off')            
            fig.delaxes(ax)
            ax.clear()
            plt.draw()
            fig.add_axes(ax)
            plt.draw()
            tracker = IndexTracker(ax, atlas_data, pixdim, plane)
            fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
            plt.show()  
            flag_color = 0            
    elif event.key == 'b':   
        # Show the names of the regions
        global cursor
        cursor = mplcursors.cursor(hover=True)
        def show_annotation(sel):
            if flag_names == 1:                
                sel.annotation.set_visible(True)
            elif flag_names == 0:                
                sel.annotation.set_visible(False)
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
        if flag_names == 0:
            print("Show region's name on")
            flag_names = 1
        elif flag_names == 1:
            print("Show region's name off")
            flag_names = 0              
    elif event.key == 'r':     
        print('Register probe 1 (purple)')
        global probe_colors                            
        probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']
        # plot  point and register all the clicked points
        def onclick_probe(event):
            global px, py
            px, py = event.xdata/pixdim, event.ydata/pixdim
            # assign global variable to access outside of function
            global coords_probe_temp_w, coords_probe_temp_g, coords_probe_temp_p, coords_probe_temp_b, coords_probe_temp_y, coords_probe_temp_o, coords_probe_temp_r,  p_probe_grid, p_probe_trans
            if probe_counter == 0:
                coords_probe_temp_w.append((px, py, tracker.ind)) 
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_w)
            elif probe_counter == 1:
                coords_probe_temp_g.append((px, py, tracker.ind))
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_g)
            elif probe_counter == 2:
                coords_probe_temp_p.append((px, py, tracker.ind))    
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_p)
            elif probe_counter == 3:
                coords_probe_temp_b.append((px, py, tracker.ind))
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_b)
            elif probe_counter == 4:
                coords_probe_temp_y.append((px, py, tracker.ind))
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_y)
            elif probe_counter == 5:
                coords_probe_temp_o.append((px, py, tracker.ind))
                p_probe.extend(ax.plot(event.xdata, event.ydata, color=probe_colors[probe_counter], marker='o', markersize=1))
                setattr(coords_probe,probe_colors[probe_counter],coords_probe_temp_o)
            fig.canvas.draw()
            return
        # Call click func
        fig.canvas.mpl_connect('button_press_event', onclick_probe) 
        def on_key2(event):            
            if event.key == 'n':
                # add a new probe
                global probe_counter
                if probe_counter+1 <len(probe_colors):
                    probe_counter +=  1                                                                                               
                    print('probe %d added (%s)' %(probe_counter+1, probe_colors[probe_counter]))
                else:
                    print('Cannot add more probes')
                    probe_counter = len(probe_colors)
                    
            elif event.key == 'c':
                print('Delete clicked probe point')
                if len(getattr(coords_probe,probe_colors[0]))!= 0:
                    if len(getattr(coords_probe,probe_colors[probe_counter])) != 0:
                        getattr(coords_probe,probe_colors[probe_counter]).pop(-1) # remove the point from the list
                        p_probe[-1].remove() # remove the point from the plot
                        fig.canvas.draw()
                        p_probe.pop(-1)                        
                    elif len(getattr(coords_probe,probe_colors[probe_counter])) == 0:
                        probe_counter -=1
                        try:
                            getattr(coords_probe,probe_colors[probe_counter]).pop(-1) # remove the point from the list
                            p_probe[-1].remove() # remove the point from the plot
                            fig.canvas.draw()                                        
                            p_probe.pop(-1)

                        except:
                            pass
# =============================================================================
#             elif event.key == 'p':
#                 print( 'Change probe' )
#                 if probe_counter-1 > 0:
#                     probe_counter -=  1                                                                                                 
#                     print('probe %d selected (%s)' %(probe_counter+1, probe_colors[probe_counter]))
#                 elif probe_counter == 0:
#                     probe_counter +=1 
#                     print('probe %d selected (%s)' %(probe_counter+1, probe_colors[probe_counter]))
# =============================================================================
        fig.canvas.mpl_connect('key_press_event', on_key2)        
    elif event.key == 'e':
        print('Probe points saved')        
        #num = input('If the probe goes over several slices insert the slice number, otherwise enter 0: ')        
        # Create and save slice, clicked probes
        
        P = save_probe_insertion(coords_probe, plane, probe_counter)        # Saving the object
# =============================================================================
#         with open(os.path.join(path_probe_insertion, num+'probes.pkl'), 'wb') as F: 
#             pickle.dump(P, F)# Create and save slice, clicked points, and image info    
# =============================================================================  
        # MAC    
        with open('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Probe_Insertion/0probes.pkl', 'wb') as F: 
            pickle.dump(P, F)# Create and save slice, clicked points, and image info 
    elif event.key == 'u':        
        print('Load probe')
        files_probe = os.listdir(path_probe_insertion)
        # =============================================================================
        # for f in files_probe:
        #     # WINDOWS
        #     P.append(pickle.load(open(os.path.join(path_probes, f), "rb")))
        # =============================================================================
        # MAC
        Pp.append(pickle.load(open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Probe_Insertion/0probes.pkl', "rb")))
        global flag
        flag = 1   
            
    elif event.key == 'w':
        global trackerp
        # if the probe if uploaded from a file
        if flag == 1:
            probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']
            # If I have several probes
            for j in range(len(probe_colors)):   
                for k in range(len(Pp)):
                    try:
                        global probe_selecter_u
                        PC = getattr(Pp[k].Probe, probe_colors[j])
                        probe_x = []
                        probe_y = []
                        probe_slice = []
                        for i in range(len(PC)):
                            probe_x.append(PC[i][0]*pixdim)
                            probe_y.append(PC[i][1]*pixdim)
                            probe_slice.append(PC[i][2])
                        unique_slice = list(OrderedDict.fromkeys(probe_slice))    
                        m, b = np.polyfit(probe_x, probe_y, 1)  
                        # display the probe in a separate window
                        fig_probe, ax_probe = plt.subplots(1, 1)  
                        trackerp = IndexTracker_pi(ax_probe, atlas_data, pixdim, plane, tracker.ind, unique_slice, probe_x, probe_y, probe_colors, probe_selecter_u, m, b)
                        fig_probe.canvas.mpl_connect('scroll_event', trackerp.onscroll)        
                        ax_probe.text(0.05, 0.95, textstr, transform=ax_probe.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
                        ax_probe.format_coord = format_coord
                        ax_probe.set_title("Probe %d viewer" %(probe_selecter_u+1))
                        plt.show()
                        mngr_probe = plt.get_current_fig_manager()
                        mngr_probe.window.setGeometry(650,250,d2*2,d1*2)                  
                        # get the probe coordinates and the region's names
                        probe_x = []
                        probe_y = []
                        probe_z = []                    
                        if Pp[k].Plane == 'c':
                            for i in range(len(PC)):
                                probe_x.append(PC[i][0]*pixdim)
                                probe_y.append(PC[i][2]*pixdim)
                                probe_z.append(PC[i][1]*pixdim)
                        elif Pp[k].Plane == 's':
                            for i in range(len(PC)):
                                probe_x.append(PC[i][2]*pixdim)
                                probe_y.append(PC[i][0]*pixdim)
                                probe_z.append(PC[i][1]*pixdim)  
                        elif Pp[k].Plane == 'h':
                            for i in range(len(PC)):        
                                probe_x.append(PC[i][0]*pixdim)
                                probe_y.append(PC[i][1]*pixdim)        
                                probe_z.append(PC[i][2]*pixdim)
                        pts = np.array((probe_x, probe_y, probe_z)).T
                        line_fit = Line.best_fit(pts)
                        # if no inclination in z direction
                        if line_fit.direction[2] == 0:
                            # line equations, to derive the send point of the line (aka probe)
                            z2 = pts[0,2]
                            x2 = pts[-1,0]
                            y2 = line_fit.point[1]+((x2-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                            deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                            deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                            # position_at_bregma_depth
                            z0 = 440*pixdim # correspond at the position of the bregma DV=0
                            x0 = pts[0,0]
                            y0 = line_fit.point[1]+((x0-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                            ML_position = (x0-246*pixdim)
                            AP_position = (y0-653*pixdim)
                            X0 = np.array([x0,y0,z0])
                            X2 = np.array([x2,y2,z2])
                            # get lenght of the probe
                            dist = np.linalg.norm(X0-X2)                
                            regions = []
                            point_along_line = []
                            s = int(math.modf(X0[0]/pixdim)[1])# starting point
                            f = int(math.modf(X2[0]/pixdim)[1]) # ending point
                            for x in range(min(s,f), max(s,f)):                        
                                y = line_fit.point[1]/pixdim+((x-line_fit.point[0]/pixdim)/line_fit.direction[0])*line_fit.direction[1]
                                z = pts[0,2]/pixdim
                                regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                                point_along_line.append([x,y,z])
                        else:
                            # line equations, to derive the send point of the line (aka probe)
                            # the last of the clicked points represent the end point of the line
                            z2 = pts[-1,2]
                            x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                            y2 = line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                            deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                            deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                            # position_at_bregma_depth
                            z0 = 440*pixdim # correspond at the position of the bregma DV=0
                            x0 = line_fit.point[0]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                            y0 = line_fit.point[1]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                            ML_position = (x0-246*pixdim)
                            AP_position = (y0-653*pixdim)
                            X0 = np.array([x0,y0,z0])
                            X2 = np.array([x2,y2,z2])
                            # get lenght of the probe
                            dist = np.linalg.norm(X0-X2)                
                            regions = []
                            point_along_line = []
                            s = int(math.modf(X0[2]/pixdim)[1])# starting point
                            f = int(math.modf(X2[2]/pixdim)[1]) # ending point
                            for z in range(min(s,f),max(s,f)):
                                x = line_fit.point[0]/pixdim+((z-line_fit.point[2]/pixdim)/line_fit.direction[2])*line_fit.direction[0]
                                y = line_fit.point[1]/pixdim+((z-line_fit.point[2]/pixdim)/line_fit.direction[2])*line_fit.direction[1]
                                regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                                point_along_line.append([x,y,z])
                        # avoid repetions and reverse the order
                        regioni = list(OrderedDict.fromkeys(regions))[::-1]  
                        if 'Clear Label' in regioni: 
                            regioni.remove('Clear Label')                         
                        num_el = []
                        for re in regioni:
                            # in the case in dont exit and then enter again the region
                            position = [i for i,x in enumerate(regions) if x==re]
                            # if there is only one point in the region
                            if len(position) == 1:
                                regional_dist = pixdim
                            else:    
                                # first point along the line in the region 
                                start = [element * pixdim for element in point_along_line[position[0]]]
                                # last point along the line in the region 
                                end = [element * pixdim for element in point_along_line[position[-1]]]
                                # length of the part of the probe in the region
                                regional_dist = distance.euclidean(start,end)  
                        # Number of electrodes in the region                
                        num_el.append(round(regional_dist/vert_el_dist)*2)                                 
                        # print insertion coordinates    
                        print('\n---Estimated probe insertion---')
                        if ML_position>0:
                            testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = R%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' %( AP_position, abs(ML_position), dist, deg_ant, deg_lat)                    
                            print('Entry position at DV = 0: AP = %.2f mm, ML = R%.2f mm' %(AP_position, abs(ML_position)))
                        else:
                            testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = L%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' %( AP_position, abs(ML_position), dist, deg_ant, deg_lat)                    
                            print('Entry position at DV = 0: AP = %.2f mm, ML = L%.2f fmm' %(AP_position, abs(ML_position)))
                        print('Insertion distance from the above position: %.2f mm' %dist)
                        print('%.2f degrees in the anterior direction' %deg_ant)
                        print('%.2f degrees in the lateral direction\n' %deg_lat)
                        # print regions and channels
                        LL = [regioni, num_el]
                        headers = [' Regions traversed', 'Channels']
                        numpy_array = np.array(LL)
                        transpose = numpy_array.T
                        transpose_list = transpose.tolist()
                        print(tabulate(transpose_list, headers, floatfmt=".2f"))
# =============================================================================
#                         print('---Regions traversed---')
#                         print(*regioni, sep = "\n")                 
# =============================================================================
                        if plane == 'c':                           
                            regioni.insert(0,'            ---Regions traversed---')
                            if len(regioni)>16: 
                                ax_probe.text(0.01, 0.26, testo, transform=ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')   
                                B = regioni[:len(regioni)//2]
                                C = regioni[len(regioni)//2:]
                                ax_probe.text(0.41, 0.26, "\n".join(B), transform=ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')                        
                                ax_probe.text(0.76, 0.26, "\n".join(C), transform=ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')
                            else:
                                ax_probe.text(0.01, 0.26, testo, transform=ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')   
                                ax_probe.text(0.51, 0.26, "\n".join(regioni), transform=ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')
                        elif plane == 's':                             
                            ax_probe.text(0.15, 0.20, testo, transform=ax_probe.transAxes, fontsize=11 ,verticalalignment='top', color = 'w')   
                            regioni.insert(0,'            ---Regions traversed---')
                            # if there are too many regions to print
                            if len(regioni)>7:
                                B = regioni[:len(regioni)//2]
                                C = regioni[len(regioni)//2:]
                                ax_probe.text(0.5, 0.25, "\n".join(B), transform=ax_probe.transAxes, fontsize=9.5 ,verticalalignment='top', color = 'w')
                                ax_probe.text(0.74, 0.25, "\n".join(C), transform=ax_probe.transAxes, fontsize=9.5 ,verticalalignment='top', color = 'w')
                            else:
                                ax_probe.text(0.51, 0.25, "\n".join(regioni), transform=ax_probe.transAxes, fontsize=11 ,verticalalignment='top', color = 'w')
                        elif plane == 'h':                    
                            
                            regioni.insert(0,'            ---Regions traversed---')
                            # if there are too many regions to print
                            if len(regioni)>7:
                                ax_probe.text(0.17, 0.22, testo, transform=ax_probe.transAxes, fontsize=8 ,verticalalignment='top', color = 'w')   
                                B = regioni[:len(regioni)//2]
                                C = regioni[len(regioni)//2:]
                                ax_probe.text(0.01, 0.15, "\n".join(B), transform=ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')
                                ax_probe.text(0.49, 0.15, "\n".join(C), transform=ax_probe.transAxes, fontsize=6.4 ,verticalalignment='top', color = 'w')
                            else:
                                ax_probe.text(0.17, 0.22, testo, transform=ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')   
                                ax_probe.text(0.17, 0.13, "\n".join(regioni), transform=ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')      
                        probe_selecter_u +=1
                    except:
                        pass
        else:                    
            try:   
                global probe_selecter
                print('\nProbe %d view mode' %(probe_selecter+1))
                L = getattr(coords_probe,probe_colors[probe_selecter])
                probe_x = []
                probe_y = []
                probe_slice = []
                for i in range(len(L)):
                    probe_x.append(L[i][0]*pixdim)
                    probe_y.append(L[i][1]*pixdim)
                    probe_slice.append(L[i][2])
                unique_slice = list(OrderedDict.fromkeys(probe_slice))    
                m, b = np.polyfit(probe_x, probe_y, 1)  
                # display the probe in a separate window
                fig_probe, ax_probe = plt.subplots(1, 1)  
                trackerp = IndexTracker_pi(ax_probe, atlas_data, pixdim, plane, tracker.ind, unique_slice, probe_x, probe_y, probe_colors, probe_selecter, m, b)
                fig_probe.canvas.mpl_connect('scroll_event', trackerp.onscroll)        
                ax_probe.text(0.05, 0.95, textstr, transform=ax_probe.transAxes, fontsize=6 ,verticalalignment='bottom', bbox=props)
                ax_probe.format_coord = format_coord
                ax_probe.set_title("Probe %d viewer" %(probe_selecter+1))
                plt.show()
                mngr_probe = plt.get_current_fig_manager()
                mngr_probe.window.setGeometry(650,250,d2*2,d1*2)    
                # get the probe coordinates and the region's names
                probe_x = []
                probe_y = []
                probe_z = []            
                if plane == 'c':
                    for i in range(len(L)):
                        probe_x.append(L[i][0]*pixdim)
                        probe_y.append(L[i][2]*pixdim)
                        probe_z.append(L[i][1]*pixdim)
                elif plane == 's':
                    for i in range(len(L)):
                        probe_x.append(L[i][2]*pixdim)
                        probe_y.append(L[i][0]*pixdim)
                        probe_z.append(L[i][1]*pixdim)  
                elif plane == 'h':
                    for i in range(len(L)):        
                        probe_x.append(L[i][0]*pixdim)
                        probe_y.append(L[i][1]*pixdim)        
                        probe_z.append(L[i][2]*pixdim)
                pts = np.array((probe_x, probe_y, probe_z)).T
                # fit the probe
                line_fit = Line.best_fit(pts)
                # if no inclination in z direction
                if line_fit.direction[2] == 0:
                    # line equations, to derive the send point of the line (aka probe)
                    z2 = pts[0,2]
                    x2 = pts[-1,0]
                    y2 = line_fit.point[1]+((x2-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                    deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                    deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                    # position_at_bregma_depth
                    z0 = 440*pixdim # correspond at the position of the bregma DV=0
                    x0 = pts[0,0]
                    y0 = line_fit.point[1]+((x0-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                    ML_position = (x0-246*pixdim)
                    AP_position = (y0-653*pixdim)
                    X0 = np.array([x0,y0,z0])
                    X2 = np.array([x2,y2,z2])
                    # get lenght of the probe
                    dist = np.linalg.norm(X0-X2)                
                    regions = []
                    point_along_line = []
                    s = int(math.modf(X0[0]/pixdim)[1])# starting point
                    f = int(math.modf(X2[0]/pixdim)[1]) # ending point
                    for x in range(min(s,f), max(s,f)):                        
                        y = line_fit.point[1]/pixdim+((x-line_fit.point[0]/pixdim)/line_fit.direction[0])*line_fit.direction[1]
                        z = pts[0,2]/pixdim
                        if int(math.modf(x)[1])>512 or int(math.modf(y)[1])>1024 or int(math.modf(z)[1])>512:
                            regions.append('Clear Label')
                        else:    
                            regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                        point_along_line.append([x,y,z])
                else:
                    # line equations, to derive the send point of the line (aka probe)
                    # the last of the clicked points represent the end point of the line
                    z2 = pts[-1,2]
                    x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                    y2 = line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                    deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                    deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                    # position_at_bregma_depth
                    z0 = 440*pixdim # correspond at the position of the bregma DV=0
                    x0 = line_fit.point[0]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                    y0 = line_fit.point[1]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                    ML_position = (x0-246*pixdim)
                    AP_position = (y0-653*pixdim)
                    X0 = np.array([x0,y0,z0])
                    X2 = np.array([x2,y2,z2])
                    # get lenght of the probe
                    dist = np.linalg.norm(X0-X2) 
                    regions = []
                    point_along_line = []
                    s = int(math.modf(X0[2]/pixdim)[1])# starting point
                    f = int(math.modf(X2[2]/pixdim)[1]) # ending point
                    for z in range(min(s,f),max(s,f)):
                        x = line_fit.point[0]/pixdim+((z-line_fit.point[2]/pixdim)/line_fit.direction[2])*line_fit.direction[0]
                        y = line_fit.point[1]/pixdim+((z-line_fit.point[2]/pixdim)/line_fit.direction[2])*line_fit.direction[1]
                        if int(math.modf(x)[1])>512 or int(math.modf(y)[1])>1024 or int(math.modf(z)[1])>512:
                            regions.append('Clear Label')
                        else:    
                            regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                        point_along_line.append([x,y,z])
                # avoid repetions and reverse the order
                regioni = list(OrderedDict.fromkeys(regions))[::-1]
                if 'Clear Label' in regioni: 
                    regioni.remove('Clear Label')
                num_el = []
                for re in regioni:
                    # in the case in dont exit and then enter again the region
                    position = [i for i,x in enumerate(regions) if x==re]
                    # if there is only one point in the region
                    if len(position) == 1:
                        regional_dist = pixdim
                    else:    
                        # first point along the line in the region 
                        start = [element * pixdim for element in point_along_line[position[0]]]
                        # last point along the line in the region 
                        end = [element * pixdim for element in point_along_line[position[-1]]]
                        # length of the part of the probe in the region
                        regional_dist = distance.euclidean(start,end)  
                    # Number of electrodes in the region                
                    num_el.append(round(regional_dist/vert_el_dist)*2)                  
                # print insertion coordinates    
                print('\n---Estimated probe insertion---')
                if ML_position>0:
                    testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = R%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' %( AP_position, abs(ML_position), dist, deg_ant, deg_lat)                    
                    print('Entry position at DV = 0: AP = %.2f mm, ML = R%.2f mm' %(AP_position, abs(ML_position)))
                else:
                    testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = L%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' %( AP_position, abs(ML_position), dist, deg_ant, deg_lat)                    
                    print('Entry position at DV = 0: AP = %.2f mm, ML = L%.2f fmm' %(AP_position, abs(ML_position)))
                print('Insertion distance from the above position: %.2f mm' %dist)
                print('%.2f degrees in the anterior direction' %deg_ant)
                print('%.2f degrees in the lateral direction\n' %deg_lat)
                # print regions and number of channels
                LL = [regioni, num_el]
                headers = [' Regions traversed', 'Channels']
                numpy_array = np.array(LL)
                transpose = numpy_array.T
                transpose_list = transpose.tolist()
                print(tabulate(transpose_list, headers, floatfmt=".2f"))
# =============================================================================
#                 print('---Regions traversed---')
#                 print(*regioni, sep = "\n")                 
# =============================================================================
                if plane == 'c':
                    
                    regioni.insert(0,'            ---Regions traversed---')
                    if len(regioni)>16:
                        ax_probe.text(0.01, 0.26, testo, transform=ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')   
                        B = regioni[:len(regioni)//2]
                        C = regioni[len(regioni)//2:]
                        ax_probe.text(0.41, 0.26, "\n".join(B), transform=ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')                        
                        ax_probe.text(0.76, 0.26, "\n".join(C), transform=ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')
                    else:
                        ax_probe.text(0.01, 0.26, testo, transform=ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')   
                        ax_probe.text(0.51, 0.26, "\n".join(regioni), transform=ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')
                elif plane == 's':
                    ax_probe.text(0.15, 0.20, testo, transform=ax_probe.transAxes, fontsize=11, verticalalignment='top', color = 'w')   
                    regioni.insert(0,'            ---Regions traversed---')
                    # if there are too many regions to print
                    if len(regioni)>7:
                        B = regioni[:len(regioni)//2]
                        C = regioni[len(regioni)//2:]
                        ax_probe.text(0.5, 0.25, "\n".join(B), transform=ax_probe.transAxes, fontsize=9.5 ,verticalalignment='top', color = 'w')
                        ax_probe.text(0.74, 0.25, "\n".join(C), transform=ax_probe.transAxes, fontsize=9.5 ,verticalalignment='top', color = 'w')
                    else:
                        ax_probe.text(0.51, 0.25, "\n".join(regioni), transform=ax_probe.transAxes, fontsize=11 ,verticalalignment='top', color = 'w')
                elif plane == 'h':                    
                    
                    regioni.insert(0,'            ---Regions traversed---')
                    # if there are too many regions to print
                    if len(regioni)>7:
                        ax_probe.text(0.17, 0.22, testo, transform=ax_probe.transAxes, fontsize=8 ,verticalalignment='top', color = 'w')   
                        B = regioni[:len(regioni)//2]
                        C = regioni[len(regioni)//2:]
                        ax_probe.text(0.01, 0.15, "\n".join(B), transform=ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')
                        ax_probe.text(0.49, 0.15, "\n".join(C), transform=ax_probe.transAxes, fontsize=6.4 ,verticalalignment='top', color = 'w')
                    else:
                        ax_probe.text(0.17, 0.22, testo, transform=ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')   
                        ax_probe.text(0.17, 0.13, "\n".join(regioni), transform=ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')                    
                probe_selecter +=1                   
            except:
                print('No more probes to visualize')
                pass        
            
fig.canvas.mpl_connect('key_press_event', on_key)












