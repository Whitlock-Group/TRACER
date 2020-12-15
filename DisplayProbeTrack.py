from __future__ import print_function

# Import libraries
import os
import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import pickle 
from skimage import measure
    
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

P = []
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

# If I have only one probe

# get the probe coordinates
probe_x = []
probe_y = []
probe_z = []
for k in range(len(P)):
    PC = getattr(P[k].Probe, probe_colors[0])
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
# fit the probe
line_fit = Line.best_fit(pts)
# line equations, to derive the starting and end point of the line (aka probe)
z1 = max(pts[:,2])
x1 = line_fit.point[0]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
y1 = line_fit.point[1]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
z2 = min(pts[:,2])
x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
y2= line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
# get the line to plot
L = vedo.Line([x1, y1, z1],[x2, y2, z2],c = 'green', lw = 2)
# clicked points to display
pp = vedo.Points(pts, c = 'green') #fast    

# load the brain regions
Edges = np.load('Edges.npy')
edges = Edges.T
coords = np.array(np.where(edges == 255))
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
    
show(mesh, pp, L , __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='white',
)






