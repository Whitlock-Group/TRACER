import numpy as np
import cv2
import os
import os.path
from os import path
import nibabel as nib
from skimage import measure
import matplotlib.pyplot as plt
    
from vedo import Volume as VedoVolume
import vedo 
from vedo import buildLUT, Sphere, show, settings
from vedo.mesh import Mesh, merge
    
Edges = np.load('Edges.npy')
coords = np.array(np.where(Edges== 255)).T

# Manage Points cloud
points = vedo.pointcloud.Points(coords)
# Create the mesh
mesh = Mesh(points)

# create some dummy data array to be associated to points
data = mesh.points()[:,2]  # pick z-coords, use them as scalar data


# build a custom LookUp Table of colors:
#               value, color, alpha
lut = buildLUT([
                #(-2, 'pink'      ),  # up to -2 is pink
                (0, 'pink'      ),  # up to 0 is pink
                (230, 'lightblue', 1),  # up to 0.4 is green with alpha=0.5
                (300, 'lightblue', 1 ),
                #( 2, 'darkblue'  ),
               ],
               vmin=0, belowColor='lightblue',
               vmax= 512, aboveColor='grey',
               nanColor='red',
               interpolate=False,
              )

mesh.cmap(lut, data)

show(mesh, __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='white',
)





