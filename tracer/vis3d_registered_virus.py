from __future__ import print_function
# Import libraries
import math 
import os
import numpy as np
import cv2
import pickle
from collections import OrderedDict
from tabulate import tabulate
    
# 3d Brain
import vedo
vedo.settings.embedWindow(backend=False, verbose=True)



from .ObjSave import probe_obj, save_probe
from .atlas_loader import AtlasLoader


def vis3d_registered_virus(atlas, virus_folder):
    """
    Purpose
    -------------
    Visualise registered virus in 3D.

    Inputs
    -------------
    atlas :
    virus_folder :

    """

    # Probe colors
    virus_colors = ['orange', 'blue', 'purple', 'yellow', 'red', 'green']

    # The virus info will be saved in a subfolder called info
    path_info = os.path.join(virus_folder, 'info')
    if not os.path.exists(path_info):
        os.mkdir(path_info)


    # get the all the files in the probe folder that are not folders
    files_virus = []
    for fname in os.listdir(virus_folder):
        if fname[-4:] != '.pkl':
            continue
        files_virus.append(fname)


    pr = probe_obj()
    PR = probe_obj()
    P = []
    color_used_t = []
    for f in sorted(files_virus):
        fnm = os.path.join(virus_folder, f)
        P.append(pickle.load(open(fnm, "rb")))
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
                pp = vedo.Points(pts, c=virus_colors[j], r=7)  #fast CHANGE COLOR HERE AND SIZE!!! ####
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
            regions.append(atlas.labels_name[np.argwhere(np.all(atlas.labels_index == atlas.segmentation_data[int(math.modf(punti[j,0])[1]),int(math.modf(punti[j,1])[1]),int(math.modf(punti[j,2])[1])], axis = 1))[0,0]])
            initials.append(atlas.labels_initial[np.argwhere(np.all(atlas.labels_index == atlas.segmentation_data[int(math.modf(punti[j,0])[1]),int(math.modf(punti[j,1])[1]),int(math.modf(punti[j,2])[1])], axis = 1))[0,0]])
            index.append(atlas.segmentation_data[int(math.modf(punti[j,0])[1]),int(math.modf(punti[j,1])[1]),int(math.modf(punti[j,2])[1])])
            ml = (punti[j,0]-246)*atlas.pixdim
            if ml > 0:
                ML.append('R '+str(abs(round(ml,2))))
            else:
                ML.append('L '+str(abs(round(ml,2))))
            AP.append((punti[j,1]-623)*atlas.pixdim)
            Z.append((punti[j,2]-440)*atlas.pixdim)
            
            # count the number of elements in each region to
        print('\nRegions of clicked points for %s probe: \n ' % color_used[i])
        LL = [regions,  initials, ML, AP,Z]
        headers = [' Regions', 'Initials', 'ML', 'AP', 'Z']
        numpy_array = np.array(LL)
        transpose = numpy_array.T
        transpose_list = transpose.tolist()
        print(tabulate(transpose_list, headers, floatfmt=".2f"))
        
        # Write and save txt file with probe info
        pn = "Virus_Info.txt"
        fnm = os.path.join(path_info, pn)
        f = open(fnm, "w+")
        f.write('Analyze virus expression: \n\n ')
        f.write(tabulate(transpose_list, headers, floatfmt=".2f"))
        f.close()

    mask_data = atlas.mask_data.transpose((2, 1, 0))
    Edges = np.empty((512, 1024, 512))
    for sl in range(0, 1024):
        Edges[:, sl, :] = cv2.Canny(np.uint8((mask_data[:, sl, :]) * 255), 100, 200)

    edges = Edges.T
    edges[:, ::2, :] = edges[:, ::2, :] * 0
    edges[:, ::5, :] = edges[:, ::5, :] * 0
    edges[::2, :, :] = edges[::2, :, :] * 0
    edges[:, :, ::2] = edges[:, :, ::2] * 0

    coords = np.array(np.where(edges == 255))
    # Manage Points cloud
    points = vedo.pointcloud.Points(coords)
    # Create the mesh
    mesh = vedo.Mesh(points)

    # create some dummy data array to be associated to points
    data = mesh.points()[:, 2]  # pick z-coords, use them as scalar data
    # build a custom LookUp Table of colors:
    lut = vedo.buildLUT([(512, 'white', 0.07), ],
                        vmin=0, belowColor='lightblue',
                        vmax=512, aboveColor='grey', nanColor='red', interpolate=False)
    mesh.cmap(lut, data)
    
    
    # plot all the probes together
    plt1 = vedo.Plotter(title='Brain viewer', size=(700, 700), pos=(250, 0))
    if n == 1:
         plt1.show(mesh, getattr(pr,color_used[0]), __doc__,
         axes=0, viewup="z", bg='black',  # change the backgrond color ###
         )
    elif n == 2:
         plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 3:
         plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 4:
         plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 5:
         plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 6:
         plt1.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(pr,color_used[5]), __doc__,
         axes=0, viewup="z", bg='black',
         )



