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

# read label file
from Readlabel import readlabel

# Labels
# Mac
labels_item = open(r"/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.label", "r")
# Windows
# labels_item = open(r"C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\Waxholm_Atlas\WHS_SD_rat_atlas_v4_beta.label", "r")
labels_index, labels_name, labels_color, labels_initial = readlabel( labels_item )   

# Segmentation
# Mac
segmentation = nib.load('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v4_beta.nii.gz')
# Windows
#segmentation = nib.load(segmentation_path)
segmentation_data = segmentation.get_fdata()

# Load the atlas
# Mac
atlas = nib.load(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v2_pack/WHS_SD_rat_T2star_v1.01.nii.gz')
# Windows
#atlas = nib.load(atlas_path)
atlas_header = atlas.header
# get pixel dimension
pixdim = atlas_header.get('pixdim')[1]



# Probe colors
probe_colors = ['green', 'purple', 'blue', 'yellow', 'orange', 'red']

# Windows
# =============================================================================
# processed_histology_folder = r'C:\Users\jacopop\Box Sync\macbook\Documents\KAVLI\histology\processed'
# path_probes = os.path.join(processed_histology_folder, 'probes')
# path_transformed = os.path.join(processed_histology_folder, 'transformations')
# =============================================================================

# Mac
processed_histology_folder = r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed'
path_probes = r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes'
path_transformed = '/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/transformations'

# get the all the files in the probe folder
files_probe = os.listdir(path_probes)
files_transformed = os.listdir(path_transformed)


pr = probe_obj()
PR = probe_obj()
P = []
color_used_t = []
# =============================================================================
# for f in files_probe:
#     # WINDOWS
#     P.append(pickle.load(open(os.path.join(path_probes, f), "rb")))
# =============================================================================
    # MAC
P.append(pickle.load(open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes/1probes.pkl', "rb")))
P.append(pickle.load(open(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed/probes/2probes.pkl', "rb")))
# LL = pickle.load(open(os.path.join(path_probes, '1probes.pkl'), "rb"))    
probe_counter = P[0].Counter

# If I have several probes
for j in range(len(probe_colors)):    
    # get the probe coordinates and the region's names
    probe_x = []
    probe_y = []
    probe_z = []
    for k in range(len(P)):
        try:
            PC = getattr(P[k].Probe, probe_colors[j])
            if P[k].Plane == 'c':
                for i in range(len(PC)):
                    probe_x.append(PC[i][0])
                    probe_y.append(P[k].Slice)
                    probe_z.append(PC[i][1])
            elif P[k].Plane == 'c':
                for i in range(len(PC)):
                    probe_x.append(P[k].Slice)
                    probe_y.append(PC[i][0])
                    probe_z.append(PC[i][1])  
            elif P[k].Plane == 'c':
                for i in range(len(PC)):        
                    probe_x.append(PC[i][0])
                    probe_y.append(PC[i][1])        
                    probe_z.append(P[k].Slice)
            pts = np.array((probe_x, probe_y, probe_z)).T 
            pp = vedo.Points(pts, c = probe_colors[j]) #fast    
            setattr(pr, probe_colors[j], pp)
            setattr(PR, probe_colors[j], pts)
            color_used_t.append(probe_colors[j])
        except:
            pass  

# get only the unique color in order                  
color_used = list(OrderedDict.fromkeys(color_used_t))
n = len(color_used)

# load the brain regions
Edges = np.load('Edges.npy')
edges = Edges.T
coords = np.array(np.where(edges == 255))

# compute and display the insertion angle for each probe
for i in range(0,n):
    regions = []
    initials = []
    index = []
    ML = []
    AP = []
    Z = []
    punti = getattr(PR, color_used[i])
    for j in range(len(punti)):
        regions.append(labels_name[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(punti[j,0])[1]),int(math.modf(punti[j,1])[1]),int(math.modf(punti[j,2])[1])], axis = 1))[0,0]])
        initials.append(labels_initial[np.argwhere(np.all(labels_index == segmentation_data[int(math.modf(punti[j,0])[1]),int(math.modf(punti[j,1])[1]),int(math.modf(punti[j,2])[1])], axis = 1))[0,0]])
        index.append(segmentation_data[int(math.modf(punti[j,0])[1]),int(math.modf(punti[j,1])[1]),int(math.modf(punti[j,2])[1])])
        ML.append((punti[j,0]-246)*pixdim)
        AP.append((punti[j,1]-653)*pixdim)
        Z.append((punti[j,0]-440)*pixdim)
        
        # count the number of elements in each region to 
    print('\nRegions of clicked points for %s probe: \n ' %color_used[i])
    LL = [regions,  initials, index, ML, AP, Z]
    headers = [' Regions', 'Initialis','Index', 'ML', 'AP', 'Z']
    numpy_array = np.array(LL)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()
    print(tabulate(transpose_list, headers))
    
        
# =============================================================================
# if ML >0:        
#     'AP=%1.4f, ML=R%1.4f, z=%1.4f'%(AP, ML, Z)
# else:
#     'AP=%1.4f, ML=L%1.4f, z=%1.4f'%(AP, ML, Z)    
# =============================================================================


# Manage Points cloud
points = vedo.pointcloud.Points(coords)
# Create the mesh
mesh = Mesh(points)

# create some dummy data array to be associated to points
data = mesh.points()[:,2]  # pick z-coords, use them as scalar data
# build a custom LookUp Table of colors:
lut = buildLUT([
                (512, 'lightgrey', 0.01 ),
               ],
               vmin=0, belowColor='lightblue',
               vmax= 512, aboveColor='grey',
               nanColor='red',
               interpolate=False,
              )
mesh.cmap(lut, data)


# plot all the probes together
if n==1:
     show(mesh, getattr(pr,color_used[0]), __doc__,
     axes=0, viewup="z", bg='white',
     )        
elif n==2:     
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), __doc__,
     axes=0, viewup="z", bg='white',
     )
elif n == 3:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), __doc__,
     axes=1, viewup="z",
     bg='white', 
     )
elif n == 4:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), __doc__,
     axes=0, viewup="z", bg='white',
     )
elif n==5:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), __doc__,
     axes=0, viewup="z", bg='white',
     )
elif n == 6:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(pr,color_used[5]), __doc__,
     axes=0, viewup="z", bg='white',
     )






