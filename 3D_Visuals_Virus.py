""" """
from __future__ import print_function

# Import libraries
import math 
import os
import os.path
from pathlib import Path
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

from ObjSave import  probe_obj, save_probe
from Tracker import IndexTracker_pi_col

# read label file
from Readlabel import readlabel


path_files = Path('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Files') 

# Paths of the atlas, segmentation and labels
## Atlas ##
atlas_folder = Path(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v2_pack')
atlas_path =  atlas_folder/'WHS_SD_rat_T2star_v1.01.nii.gz'
atlas = nib.load(atlas_path)
atlas_header = atlas.header
pixdim = atlas_header.get('pixdim')[1]
#atlas_data = atlas.get_fdata()
atlas_data = np.load(path_files/'atlas_data_masked.npy')
## Mask ##
mask_folder = Path(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas')
mask_path = mask_folder/'WHS_SD_rat_brainmask_v1.01.nii.gz'
mask = nib.load(mask_path)
mask_data = mask.get_fdata()[:,:,:,0].transpose((2,1,0))
## Segmentation ##
segmentation_folder = Path(r'/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas')
segmentation_path = segmentation_folder/'WHS_SD_rat_atlas_v3.nii.gz'
segmentation = nib.load(segmentation_path)
segmentation_data = segmentation.get_fdata()
## Labels ##
labels_item = open(r"/Users/jacopop/Box Sync/macbook/Documents/KAVLI/Waxholm_Atlas/WHS_SD_rat_atlas_v3.label", "r")
labels_index, labels_name, labels_color, labels_initial = readlabel( labels_item ) 

# Probe colors
virus_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']

processed_histology_folder = Path('/Users/jacopop/Box Sync/macbook/Documents/KAVLI/histology/processed')
path_virus =processed_histology_folder/'virus'


# The virus info will be saved in a subfolder called info
path_info = path_virus/'info'
if not os.path.exists(os.path.join(path_virus, 'info')):
    os.mkdir(path_info)


# get the all the files in the probe folder that are not folders
files_virus = []
for fname in os.listdir(path_virus):
    pathcheck = os.path.join(path_virus, fname)
    if fname.startswith('.'):
        continue
    if os.path.isdir(pathcheck):
        continue
    files_virus.append(fname)


pr = probe_obj()
PR = probe_obj()
P = []
color_used_t = []
for f in sorted(files_virus):
    P.append(pickle.load(open(path_virus/f , "rb")))    
# probe_counter = P[0].Counter

# If I have several probes
for j in range(len(virus_colors)):    
    # get the probe coordinates and the region's names
    virus_x = []
    virus_y = []
    virus_z = []
    for k in range(len(P)):
        try:
            PC = getattr(P[k].Probe, virus_colors[j])
            if P[k].Plane == 'c':
                for i in range(len(PC)):
                    virus_x.append(PC[i][0])
                    virus_y.append(P[k].Slice)
                    virus_z.append(PC[i][1])
            elif P[k].Plane == 's':
                for i in range(len(PC)):
                    virus_x.append(P[k].Slice)
                    virus_y.append(PC[i][0])
                    virus_z.append(PC[i][1])  
            elif P[k].Plane == 'h':
                for i in range(len(PC)):        
                    virus_x.append(PC[i][0])
                    virus_y.append(PC[i][1])        
                    virus_z.append(P[k].Slice)
            pts = np.array((virus_x, virus_y, virus_z)).T 
            pp = vedo.Points(pts, c = virus_colors[j], r = 7) #fast    #### CHANGE COLOR HERE AND SIZE!!! ####
            setattr(pr, virus_colors[j], pp)
            setattr(PR, virus_colors[j], pts)
            color_used_t.append(virus_colors[j])
        except:
            pass  

# get only the unique color in order                  
color_used = list(OrderedDict.fromkeys(color_used_t))
n = len(color_used)

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
        ml = (punti[j,0]-246)*pixdim
        if ml > 0:
            ML.append('R '+str(abs(round(ml,2))))
        else:
            ML.append('L '+str(abs(round(ml,2))))
        AP.append((punti[j,1]-653)*pixdim)
        Z.append((punti[j,2]-440)*pixdim)
        
        # count the number of elements in each region to 
    print('\nRegions of clicked points for %s probe: \n ' %color_used[i])
    LL = [regions,  initials, ML, AP,Z]
    headers = [' Regions', 'Initials', 'ML', 'AP', 'Z']
    numpy_array = np.array(LL)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()
    print(tabulate(transpose_list, headers, floatfmt=".2f"))
    
    # Write and save txt file with probe info
    pn = "Virus_Info.txt"
    f = open(path_info/pn,"w+")
    f.write('Analyze virus expression: \n\n ')
    f.write(tabulate(transpose_list, headers, floatfmt=".2f"))
    f.close() 

# load the brain regions
Edges = np.empty((512,1024,512))
for sl in range(0,1024):
    Edges[:,sl,:] = cv2.Canny(np.uint8((mask_data[:,sl,:])*255),100,200)  

edges = Edges.T
edges[:,::2,:] = edges[:,::2,:]*0
edges[:,::5,:] = edges[:,::5,:]*0
edges[::2,:,:] = edges[::2,:,:]*0
edges[:,:,::2] = edges[:,:,::2]*0

coords = np.array(np.where(edges == 255))
# Manage Points cloud
points = vedo.pointcloud.Points(coords)
# Create the mesh
mesh = Mesh(points)

# create some dummy data array to be associated to points
data = mesh.points()[:,2]  # pick z-coords, use them as scalar data
# build a custom LookUp Table of colors:
lut = buildLUT([
                (512, 'white', 0.1 ),  ### Change brain color and transparency ####
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
     axes=0, viewup="z", bg='black',  ### change the backgrond color ###
     )        
elif n==2:     
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), __doc__,
     axes=0, viewup="z", bg='black',
     )
elif n == 3:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), __doc__,
     axes=0, viewup="z", bg='black', 
     )
elif n == 4:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), __doc__,
     axes=0, viewup="z", bg='black',
     )
elif n==5:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), __doc__,
     axes=0, viewup="z", bg='black',
     )
elif n == 6:
     show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(pr,color_used[5]), __doc__,
     axes=0, viewup="z", bg='black',
     )






