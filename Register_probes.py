from __future__ import print_function

# Import libraries
import os
import os.path
from os import path
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from PIL import Image
import cv2
import math 
import mplcursors
from nilearn.image import resample_img
from skimage import io, transform
import pickle 
from six.moves import input 
from skspatial.objects import Line

# Functions defined in separate files
# read label file
from Readlabel import readlabel
# from readlabel_customized import readlabel_c
# Allow to navigate in the atlas
from Tracker import IndexTracker, IndexTracker_g, IndexTracker_p
# create objects to svae transformations and probes
from ObjSave import  save_transform, probe_obj, save_probe

  
path_files = Path('/Users/admin/RatBrain')      

# Directory of the processed histology
processed_histology_folder = Path('/Users/admin/RatBrain/histology/processed')

# get the all the files in the probe folder and load the his

img_hist_temp = []
img_hist = []
names = []
for fname in os.listdir(processed_histology_folder):
    pathcheck = os.path.join(processed_histology_folder, fname)
    if fname.startswith('.'):
        continue
    if os.path.isdir(pathcheck):
        continue
    img_hist_temp.append(Image.open(processed_histology_folder/fname).copy())
    img_hist.append(cv2.imread(str(processed_histology_folder/fname),cv2.IMREAD_GRAYSCALE))
    names.append(fname)


# Paths of the atlas, segmentation and labels
## Atlas ##
atlas_folder = Path(r'/Users/admin/RatBrain/Waxholm_Atlas/WHS_SD_rat_atlas_v2_pack')
atlas_path =  atlas_folder/'WHS_SD_rat_T2star_v1.01.nii.gz'
atlas = nib.load(atlas_path)
atlas_header = atlas.header
pixdim = atlas_header.get('pixdim')[1]
#atlas_data = atlas.get_fdata()
atlas_data = np.load(path_files/'atlas_data_masked.npy')
## Mask ##
mask_folder = Path(r'/Users/admin/RatBrain/Waxholm_Atlas')
mask_path = mask_folder/'WHS_SD_rat_brainmask_v1.01.nii.gz'
mask = nib.load(mask_path)
mask_data = mask.get_fdata()[:,:,:,0].transpose((2,1,0))
## Segmentation ##
segmentation_folder = Path(r'/Users/admin/RatBrain/Waxholm_Atlas')
segmentation_path = segmentation_folder/'WHS_SD_rat_atlas_v4_beta.nii.gz'
segmentation = nib.load(segmentation_path)
segmentation_data = segmentation.get_fdata()
## Labels ##
labels_item = open(r"/Users/admin/RatBrain/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.label", "r")
labels_index, labels_name, labels_color, labels_initial = readlabel( labels_item ) 

# Atlas in RGB colors according with the label file
cv_plot = np.load(path_files/'cv_plot.npy')/255

# get the edges of the colors defined in the label
Edges = np.load(path_files/'Edges.npy')

# Insert the plane of interest
plane = str(input('Select the plane coronal (c), sagittal (s), or horizontal (h): ')).lower()
# Check if the input is correct
while plane != 'c' and plane != 's' and plane != 'h':
    print('Error: Wrong plane name \n')
    plane = str(input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')).lower()

jj = 0
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
        ML = x - 246*pixdim
        Z = y - 440*pixdim
        if ML >0:        
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, abs(ML), Z)    
    ax.format_coord = format_coord
elif plane == 's':
    # dimensions
    d1 = 1024 
    d2 = 512    
    d3 = 512
    # display the coordinates relative to the bregma when hovering with the cursor
    def format_coord(x, y):
        AP = x - 653*pixdim
        ML = tracker.ind*pixdim - 246*pixdim
        Z = y - 440*pixdim
        if ML >0:        
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, abs(ML), Z)    
    ax.format_coord = format_coord
elif plane == 'h':
    # dimensions
    d1 = 512
    d2 = 1024
    d3 = 512    
    # display the coordinates relative to the bregma when hovering with the cursor
    def format_coord(x, y):
        AP = y - 653*pixdim
        ML = x - 246*pixdim        
        Z = tracker.ind*pixdim - 440*pixdim
        if ML >0:        
            return 'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, abs(ML), Z)
        else:
            return 'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, abs(ML), Z)    
    ax.format_coord = format_coord
plt.show()    
# Fix size and location of the figure window
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(800,300,d1,d2)      

# get the pixel dimension
dpi_hist = img_hist_temp[jj].info['dpi'][1]
pixdim_hist = 25.4/dpi_hist # 1 inch = 25,4 mm
# Show the HISTOLOGY
# Set up figure
fig_hist, ax_hist = plt.subplots(1, 1, figsize=(float(img_hist[jj].shape[1])/dpi_hist,float(img_hist[jj].shape[0])/dpi_hist))
ax_hist.set_title("Histology viewer")
# Show the histology image  
ax_hist.imshow(img_hist_temp[jj], extent=[0, img_hist[jj].shape[1]*pixdim_hist, img_hist[jj].shape[0]*pixdim_hist, 0])
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
mngr_hist.window.setGeometry(150,300,d1,d2)
        
print('\nControls: \n')
print('--------------------------- \n')
print('t: activate mode where clicks are logged for transform \n')
print('d: delete most recent transform point \n')
print('h: overlay of current histology slice \n')
print('x: save transform and current atlas location \n')
print('u: load saved transform and atlas location \n')   ##
print("b: scroll through region's boundaries \n")
print('a: visualization of boundaries \n')
print('g: activate gridlines \n')
print('v: activate color atlas mode \n\n')
print('r: activate mode where clicks are logged for probe \n')
print('c: delete most recent probe point \n')
print('n: add a new probe \n')
print('e: save current probe \n')
print('w: enable probe viewer mode for current probe  \n')
print('--------------------------- \n')
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
rednum_atlas = []
rednum_hist = []
# List of probe points
p_probe_trans = []
p_probe_grid = []
# Initialize probe counter and selecter
probe_counter = 0
probe_selecter = 0
flag = 0

# Set up the figure    
plt.ioff()
fig_trans, ax_trans = plt.subplots(1, 1)
mngr_trans = plt.get_current_fig_manager()
mngr_trans.window.setGeometry(200,350,d2,d1)  

# Reaction to key pressed
def on_key(event):                   
    global img_warped, ax_grid, fig_grid
    global fig_trans, ax_trans, img2
    global fig, ax, d1, d2, tracker, textstr, dpi_atl, props, mngr, format_coord, jj
    global fig_hist, ax_hist, pixdim_hist, dpi_hist, mngr_hist
    global coords_atlas, redp_atlas, rednum_atlas, cid_atlas
    global coords_hist, redp_hist, rednum_hist, cid_hist
    if event.key == 't': 
        print('\nRegister %s' %os.path.splitext(names[jj])[0])
        print('Select at least 4 points in the same order in both figures')    
        global clicka         
        clicka = 0
        # ATLAS
        # Mouse click function to store coordinates. Leave a red dot when a point is clicked
        def onclick(event):
            global ix, iy, clicka
            ix, iy = event.xdata/pixdim, event.ydata/pixdim
            clicka +=1
            # assign global variable to access outside of function            
            coords_atlas.append((ix, iy))
            redp_atlas.extend(plt.plot(event.xdata, event.ydata, 'ro',markersize=2))
            rednum_atlas.append(plt.text(event.xdata, event.ydata, clicka, fontsize = 8, color = 'red'))
            fig.canvas.draw()
            return
        # Call click func
        cid_atlas = fig.canvas.mpl_connect('button_press_event', onclick)  
        global clickh
        clickh = 0
        # HISTOLOGY  
        # Mouse click function to store coordinates. Leave a red dot when a point is clicked      
        def onclick_hist(event):
            global xh, yh, clickh
            xh, yh = event.xdata/pixdim_hist, event.ydata/pixdim_hist
            clickh +=1
            # assign global variable to access outside of function            
            coords_hist.append((xh, yh))
            redp_hist.extend(plt.plot(event.xdata, event.ydata, 'ro',markersize=2))
            rednum_hist.append(plt.text(event.xdata, event.ydata, clickh, fontsize = 8, color = 'red'))
            fig_hist.canvas.draw()
            return
        # Call click func
        cid_hist = fig_hist.canvas.mpl_connect('button_press_event', onclick_hist)
    
    elif event.key == 'u':     
        # if the histology and atlas have been already overlayed in a previous study it is possible to load it and keep working from that stage
        # and start recording the probes
        print('\nLoad image and slice')
        path_transformed = processed_histology_folder/'transformations'
        # name of the file
        im_n = input('Image name: ')
        image_name = im_n+'.pkl'
        # load the file
        IM = pickle.load(open(path_transformed/image_name, "rb"))
        tracker.ind = IM.Slice
        img2 = IM.Transform
        img_warped = IM.Transform_withoulines
        # open the wreaped figure
        ax_trans.imshow(img_warped, origin="lower", extent=[0, d1*pixdim, 0, d2*pixdim] )
        ax_trans.set_title("Histology adapted to atlas")
        plt.show()
        # open the overlayed figure
        fig_grid, ax_grid = plt.subplots(1, 1)
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
        mngr_grid.window.setGeometry(200,350,d2,d1)   
        global flag
        flag = 1            
        print('Image loaded')
        
    elif event.key == 'h':
        print('Transform histology to adpat to the atlas')
        # get the projective transformation from the set of clicked points  
        t = transform.ProjectiveTransform()
        t.estimate(np.float32(coords_atlas),np.float32(coords_hist))
        img_hist_tempo = np.asanyarray(img_hist_temp[jj])
        img_warped = transform.warp(img_hist_tempo, t, output_shape = (d1,d2), order=1, clip=False)#, mode='constant',cval=float('nan'))
        # Show the  transformed figure  
        #fig_trans, ax_trans = plt.subplots(1, 1)#, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl))        
        ax_trans.imshow(img_warped, origin="lower", extent=[0, d1*pixdim, 0, d2*pixdim] )
        ax_trans.set_title("Histology adapted to atlas")
        fig_trans.canvas.draw()
        plt.show() 
    
    elif event.key == 'b':
        print('Simple overlay to scroll through brain regions')
        # SIMPLE OVERLAY
        # here you can scroll the atlas grid
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
        mngr_grid.window.setGeometry(850,350,d2,d1)   
        
    elif event.key == 'd':
        print('Delete clicked point')
        try:
            coords_atlas.pop(-1) # remove the point from the list
            rednum_atlas[-1].remove() # remove the numbers from the plot
            redp_atlas[-1].remove() # remove the point from the plot
            fig.canvas.draw()
            rednum_atlas.pop(-1)
            redp_atlas.pop(-1)
            clicka -= 1
        except:
            pass            
        try:
            rednum_hist[-1].remove() # remove the numbers from the plot
            coords_hist.pop(-1) # remove the point from the list        
            redp_hist[-1].remove() # remove the point from the plot
            fig_hist.canvas.draw()
            redp_hist.pop(-1)
            rednum_hist.pop(-1)
            clickh -= 1
        except:
            pass
    elif event.key == 'x':
        print('Save image and slice')            
        image_n = input('Image name: ')
        image_name = image_n+'.pkl'
        # The Transformed images will be saved in a subfolder of process histology called transformations
        path_transformed = processed_histology_folder/'transformations'
        if not path.exists(os.path.join(processed_histology_folder, 'transformations')):
            os.mkdir(path_transformed)
        # Create and save slice, clicked points, and image info                                 
        S = save_transform(tracker.ind, [coords_hist, coords_atlas], img2, img_warped)        # Saving the object
        with open(os.path.join(path_transformed, image_name), 'wb') as f: 
            pickle.dump(S, f)
        # Save the images
        fig_name = image_name+'_Transformed_withoutlines.jpeg'
        fig_trans.savefig(os.path.join(path_transformed, fig_name))
        print('Image saved')                   
        
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
        print('Register probe 1 (purple)')
        try:
            plt.close(fig_g)
        except:
            pass
        try: 
            plt.close(fig_color) +m Æsmall
        except:
            pass
        # probes have different colors 
        global probe_colors, cid_trans                            
        probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']
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
        cid_trans = fig_trans.canvas.mpl_connect('button_press_event', onclick_probe) 
        
        def on_key2(event):            
            if event.key == 'n':
                # add a new probe, the function in defined in onclick_probe
                global probe_counter
                if probe_counter+1 <len(probe_colors):
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
        fig_trans.canvas.mpl_connect('key_press_event', on_key2)                  
    elif event.key == 'w':
        try:   
            global probe_selecter, fig_probe
            print('probe %d view mode' %(probe_selecter+1))
            L = getattr(coords_probe,probe_colors[probe_selecter])
            probe_x = []
            probe_y = []
            for i in range(len(L)):
                probe_x.append(L[i][0]*pixdim)
                probe_y.append(L[i][1]*pixdim)
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
            mngr_probe = plt.get_current_fig_manager()
            mngr_probe.window.setGeometry(900,400,d2,d1)    
            # plot the clicked points
            plt.scatter(probe_x, probe_y, color=probe_colors[probe_selecter], s=2)#, marker='o', markersize=1)            
            # plot the probe
            pts = np.array((probe_x, probe_y)).T
            # fit the probe
            line_fit = Line.best_fit(pts)
            if line_fit.direction[0] == 0:
                plt.plot(np.array(sorted(probe_x)), np.array(sorted(probe_y)),color=probe_colors[probe_selecter], linestyle='dashed', linewidth=0.8)
            else:
                m, b = np.polyfit(probe_x, probe_y, 1)
                plt.plot(np.array(sorted(probe_x)), m*np.array(sorted(probe_x)) + b,color=probe_colors[probe_selecter], linestyle='dashed', linewidth=0.8)
            probe_selecter +=1        
        except:
            print('No more probes to visualize')
            pass
        
    elif event.key == 'e':  
        # When saving probes use names in increasing order (Alphabetical or numerical) from the one with the first clicked point to the one with the last clicked point. 
        # Since the order of the clicked points determin the starting and ending of the probe
        path_probes = processed_histology_folder/'probes'
        if not path.exists(os.path.join(processed_histology_folder, 'probes')):
            os.mkdir(path_probes)
        # Create and save slice, clicked probes
        P = save_probe(tracker.ind, coords_probe, plane, probe_counter)        # Saving the object
        probe_n = input('Probe name: ')
        probe_name = probe_n+'_probes.pkl'
        # MAC    
        with open(path_probes/probe_name, 'wb') as F: 
            pickle.dump(P, F)# Create and save slice, clicked points, and image info 
        print('Probe points saved')
        
        try:
            # Close figures and clear variables
            plt.close(fig_grid)
            try: 
                plt.close(fig_probe)
            except:
                pass
            for i in range(len(coords_atlas)):
                coords_atlas.pop(-1) # remove the point from the list
                rednum_atlas[-1].remove() # remove the numbers from the plot
                redp_atlas[-1].remove() # remove the point from the plot
                fig.canvas.draw()
                redp_atlas.pop(-1)
                rednum_atlas.pop(-1)
                rednum_hist[-1].remove() # remove the numbers from the plot
                coords_hist.pop(-1) # remove the point from the list        
                redp_hist[-1].remove() # remove the point from the plot
                fig_hist.canvas.draw()
                redp_hist.pop(-1)
                rednum_hist.pop(-1)
                
            for j in range(len(probe_colors)):
                try:
                    for i in range(len(getattr(coords_probe,probe_colors[j]))):
                        getattr(coords_probe,probe_colors[j]).pop(-1) # remove the point from the list
                        p_probe_trans[-1].remove() # remove the point from the plot
                        fig_trans.canvas.draw()
                        p_probe_trans.pop(-1)                                  
                        p_probe_grid.pop(-1)
                except:
                   pass
            del clicka, clickh
           
            probe_selecter = 0
            # Disconnnect the registartion of the clicked points to avoid double events
            jj +=1 
            fig.canvas.mpl_disconnect(cid_atlas)
            fig_hist.canvas.mpl_disconnect(cid_hist)  
            fig_trans.canvas.mpl_disconnect(cid_trans)  
                
            #OPEN A NEW HISTOLOGY FOR NEXT REGISTRATION
            # get the pixel dimension
            dpi_hist = img_hist_temp[jj].info['dpi'][1]            
            pixdim_hist = 25.4/dpi_hist # 1 inch = 25,4 mm
            # Show the HISTOLOGY
            # Set up figure
            #fig_hist, ax_hist = plt.subplots(1, 1, figsize=(float(img_hist[jj].shape[1])/dpi_hist,float(img_hist[jj].shape[0])/dpi_hist))
            ax_hist.set_title("Histology viewer")
            # Show the histology image  
            ax_hist.imshow(img_hist_temp[jj], extent=[0, img_hist[jj].shape[1]*pixdim_hist, img_hist[jj].shape[0]*pixdim_hist, 0])
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
            fig_hist.canvas.draw()
            plt.show()
            # Fix size and location of the figure window
            mngr_hist = plt.get_current_fig_manager()
            mngr_hist.window.setGeometry(150,300,d1,d2)        
        except:
            print('\nNo more histology slice to register')
            plt.close('all')
            
          
fig.canvas.mpl_connect('key_press_event', on_key)
fig_hist.canvas.mpl_connect('key_press_event', on_key)
fig_trans.canvas.mpl_connect('key_press_event', on_key)
