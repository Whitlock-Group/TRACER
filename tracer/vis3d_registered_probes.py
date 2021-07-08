# Import libraries
import sys
sys.path.extend(['/Users/jingyig/Work/Kavli/PyCode/vitlab/github_code/TRACER'])
import math 
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pickle
from collections import OrderedDict, Counter
from tabulate import tabulate
from scipy.spatial import distance
    
# 3d Brain
import vedo
vedo.settings.embedWindow(backend=False, verbose=True)
# from vedo import buildLUT, Sphere, show, settings
# from vedo.mesh import Mesh, merge
# from vedo import *

# fit the probe
from skspatial.objects import Line

from ObjSave import probe_obj, save_probe
from Tracker import IndexTracker_pi_col
from Atlas_Loader import AtlasLoader







def vis3d_probe_track(atlas, probe_folder):
    """
    Purpose
    -------------
    Read the atlas label file.

    Inputs
    -------------
    file :

    Outputs
    -------------
    A list contains ...

    """
    if not os.path.exists(probe_folder):
        raise Exception('Please give the correct folder.')

    # PROBE
    max_probe_length = 10  # maximum length of probe shank is 10mm
    probe_widht = 0.07
    probe_thickness = 0.024
    probe_tip_length = 0.175
    total_electrodes = 960  # total number of recording sites
    electrode = 0.012  # Electrode size is 12x12 micron
    vert_el_dist = 0.02   # There are 2 electrodes every 0.02 mm

    # Probe colors
    probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']
    
    # The modified images will be saved in a subfolder called processed
    path_info = os.path.join(probe_folder, 'info')
    if not os.path.exists(path_info):
        os.mkdir(path_info)

    # get the all the files in the probe folder that are not folders
    files_probe = []
    for fname in os.listdir(probe_folder):
        if fname[-4:] != '.pkl':
            continue
        files_probe.append(fname)



    L = probe_obj()
    LINE_FIT = probe_obj()
    POINTS = probe_obj()
    pr = probe_obj()
    xyz = probe_obj()
    LENGTH = probe_obj()
    P = []

    color_used_t = []
    for f in sorted(files_probe):
        c_file = open(os.path.join(probe_folder, f), 'rb')
        da_data = pickle.load(c_file)
        P.append(da_data)
        c_file.close()
    
        
    # probe_counter = P[0].Counter
    
    # If I have several probes
    for j in range(len(probe_colors)):
        print(('j', j))
        # get the probe coordinates and the region's names
        probe_x = []
        probe_y = []
        probe_z = []
        # Needed to plot colors and probe
        p_x = []
        p_y = []
        for k in range(len(P)):
            try:
                PC = getattr(P[k].Probe, probe_colors[j])
                if P[k].Plane == 'c':
                    for i in range(len(PC)):
                        probe_x.append(PC[i][0])
                        probe_y.append(P[k].Slice)
                        probe_z.append(PC[i][1])
                        # Needed to plot colors and probe
                        p_x.append(PC[i][0] * atlas.pixdim)
                        p_y.append(PC[i][1] * atlas.pixdim)
                elif P[k].Plane == 's':
                    for i in range(len(PC)):
                        probe_x.append(P[k].Slice)
                        probe_y.append(PC[i][0])
                        probe_z.append(PC[i][1])
                        # Needed to plot colors and probe
                        p_x.append(PC[i][0] * atlas.pixdim)
                        p_y.append(PC[i][1] * atlas.pixdim)
                elif P[k].Plane == 'h':
                    for i in range(len(PC)):
                        probe_x.append(PC[i][0])
                        probe_y.append(PC[i][1])
                        probe_z.append(P[k].Slice)
                        # Needed to plot colors and probe
                        p_x.append(PC[i][0] * atlas.pixdim)
                        p_y.append(PC[i][1] * atlas.pixdim)
                pts = np.array((probe_x, probe_y, probe_z)).T
                # fit the probe
                line_fit = Line.best_fit(pts)
                # if no inclination in z direction
                if line_fit.direction[2] == 0:
                    # line equations, to derive the starting and end point of the line (aka probe)
                    # it is impartant that the first point clicked is the most external one, and the last correspond to the end of the probe
                    z1 = pts[0,2]
                    x1 = pts[0,0]
                    y1 = line_fit.point[1]+((x1-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                    z2 = pts[0,2]
                    x2 = x2 = pts[-1,0]
                    y2 = line_fit.point[1]+((x2-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                else:
                    # line equations, to derive the starting and end point of the line (aka probe)
                    # it is important that the first point clicked is the most external one, and the last correspond to the end of the probe
                    z1 = pts[0,2]
                    x1 = line_fit.point[0]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                    y1 = line_fit.point[1]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                    z2 = pts[-1,2]
                    x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                    y2 = line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                    # end point minus tip length
                    dq = (probe_tip_length)**2
                    div = 1 + (line_fit.direction[0]/line_fit.direction[2])**2 + (line_fit.direction[1]/line_fit.direction[2])**2
                    zt = z2 + math.sqrt(dq/div)
                    xt = line_fit.point[0]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                    yt = line_fit.point[1]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                    # get lenght of the probe
                    dista = np.linalg.norm(np.array([x1,y1,z1])-np.array([x2,y2,z2]))
                    dist_check = np.linalg.norm(np.array([x1,y1,z1])-np.array([xt,yt,zt]))
                    # check kthat the new end point is before the end of the tip and not after
                    if dist_check > dista:
                        zt = z2 - math.sqrt(dq/div)
                        xt = line_fit.point[0]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                        yt = line_fit.point[1]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
    
                dist = distance.euclidean((x1, y1, z1), (x2, y2, z2)) # probe length
                dist_mm = dist * atlas.pixdim  # probe length in mm
                # get the line to plot
                l = vedo.Line([x1, y1, z1],[x2, y2, z2],c=probe_colors[j], lw=2)
                # clicked points to display
                pp = vedo.Points(pts, c=probe_colors[j])  # fast
                setattr(xyz, probe_colors[j], [[x1, y1, z1], [xt, yt, zt]])
                setattr(LENGTH, probe_colors[j], [dist_mm, dist])
                setattr(pr, probe_colors[j], pp)
                setattr(L, probe_colors[j], l)
                setattr(LINE_FIT, probe_colors[j], line_fit)
                setattr(POINTS, probe_colors[j], [p_x, p_y])
                color_used_t.append(probe_colors[j])
            except:
                print('passs')
                pass
    
    print(1)
    # get only the unique color in order
    color_used = list(OrderedDict.fromkeys(color_used_t))
    n = len(color_used)
    print(('n', n))


    # load the brain regions
    mask_data = atlas.mask_data.transpose((2,1,0))
    Edges = np.empty((512, 1024, 512))
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
    mesh = vedo.mesh.Mesh(points)
    print(2)
    # create some dummy data array to be associated to points
    data = mesh.points()[:,2]  # pick z-coords, use them as scalar data
    # build a custom LookUp Table of colors:
    lut = vedo.buildLUT([
                    (512, 'white', 0.07 ),
                   ],
                   vmin=0, belowColor='lightblue',
                   vmax= 512, aboveColor='grey',
                   nanColor='red',
                   interpolate=False,
                  )
    mesh.cmap(lut, data)
    
    print(3)
    # Ed = np.load(path_files/'Edges.npy')
    # To plot the probe with colors
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    # compute and display the insertion angle for each probe
    for i in range(0,n):
        line_fit = getattr(LINE_FIT, color_used[i])
        deg_lat = math.degrees(math.atan(line_fit.direction[0]))
        deg_ant = math.degrees(math.atan(line_fit.direction[1]))
        Length = getattr(LENGTH, color_used[i])
        print('\n\nAnalyze %s probe: \n ' % color_used[i])
        print('Probe length: %.2f mm \n' % Length[0])
        print('Estimated %s probe insertion angle: ' % color_used[i])
        print('%.2f degrees in the anterior direction' % deg_ant)
        print('%.2f degrees in the lateral direction\n' % deg_lat)
        
        # Get the brain regions traversed by the probe
        X1 = getattr(xyz, color_used[i])[0]
        X2 = getattr(xyz, color_used[i])[1]
    
        regions = []
        colori = []
        initials = []
        index = []
        channels = []
        point_along_line = []
        if line_fit.direction[2] == 0:
            s = int(math.modf(X1[0])[1])  # starting point
            f = int(math.modf(X2[0])[1])  # ending point
            for x in range(min(s,f), max(s,f)):
                y = line_fit.point[1]+((x-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                z = pts[0,2]
                point_along_line.append([x,y,z])
                regions.append(atlas.labels_name[np.argwhere(np.all(atlas.labels_index == atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                colori.append(atlas.labels_color[np.argwhere(np.all(atlas.labels_index == atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                initials.append(atlas.labels_initial[np.argwhere(np.all(atlas.labels_index == atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                #index.append(segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])])
                #channels.append(0)
        else:
            s = int(math.modf(X1[2])[1])  # starting point
            f = int(math.modf(X2[2])[1])  # ending point
            for z in range(min(s,f),max(s,f)):
                x = line_fit.point[0]+((z-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                y = line_fit.point[1]+((z-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                point_along_line.append([x,y,z])
                regions.append(atlas.labels_name[np.argwhere(np.all(atlas.labels_index == atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                colori.append(atlas.labels_color[np.argwhere(np.all(atlas.labels_index == atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                initials.append(atlas.labels_initial[np.argwhere(np.all(atlas.labels_index == atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis = 1))[0,0]])
                #index.append(segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])])
                #channels.append(0)
        # get lenght of the probe
        if Length[0] > max_probe_length + probe_tip_length:
            print('ERROR: probe %d (%s) exceed the maximum probe length (10mm)!\n' % (i+1, color_used[i]))
        recording_probe = Length[0] - probe_tip_length  # subtract the tip of the probe to consider only the part with electrodes
        electrodes_inside = round((recording_probe/vert_el_dist)*2)  # 2 electrodes avery 20 micron
        
        # count the number of elements in each region to
        counter_regions = dict(Counter(regions))
        regioni = list(OrderedDict.fromkeys(regions))
        iniziali = list(OrderedDict.fromkeys(initials))
        # remove clear label
        if 'Clear Label' in regioni:
            indice = regioni.index('Clear Label')
            regioni.pop(indice)
            iniziali.pop(indice)
        cc = 0
        jj = 0
        num_el = []
        indici = []
        for re in regioni:
            # store the index o the region to print only the color of the regions of interest
            indici.append(atlas.labels_name.index(re))
            # in the case in dont exit and then enter again the region
            position = [i for i,x in enumerate(regions) if x == re]
            # if there is only one point in the region
            if len(position) == 1:
                regional_dist = atlas.pixdim
            else:
                # first point along the line in the region
                start = [element * atlas.pixdim for element in point_along_line[position[0]]]
                # last point along the line in the region
                end = [element * atlas.pixdim for element in point_along_line[position[-1]]]
                # length of the part of the probe in the region
                regional_dist = distance.euclidean(start,end)
            # Number of electrodes in the region
            num_el.append(round(regional_dist/vert_el_dist)*2)
            #print(re)
            # proportion of the probe in the given region
            dist_prop = Length[1]/len(regioni)
            color_prop = atlas.labels_color[np.argwhere(np.array(atlas.labels_name) == re)]
            # length of the longest probe
            m = []
            for k in range(0,n):
                mt = getattr(LENGTH, color_used[k])
                m.append(mt[1])
            max_val = max(m)
            print(max_val)
            # plot the probe with the colors of the region traversed
            ax1.add_patch(patches.Rectangle((100*i+20, cc), 17, dist_prop, color=color_prop[0][0]/255))
            ax1.text(100*i, (max_val + 10), 'Probe %d\n(%s)' % (i+1, color_used[i]), color=color_used[i], fontsize=9, fontweight='bold')
            if len(iniziali[jj]) > 7:
                ax1.text(100*i-12, cc+2, '%s-\n%s' % (iniziali[jj][0:5], iniziali[jj][6:]), fontsize=5.6)
            else:
                ax1.text(100*i-12, cc+4, '%s' % (iniziali[jj]), fontsize=6)
            ax1.text(100*i+48, cc+4, '%d' % (num_el[jj]), fontsize=6.5)
            jj +=1
            cc = dist_prop + cc
            del regional_dist, position
            
        LL = [regioni,  iniziali, num_el]
        headers = [' Regions traversed', 'Initials', 'Channels']
        numpy_array = np.array(LL)
        transpose = numpy_array.T
        transpose_list = transpose.tolist()
        transpose_list.reverse()
        print(tabulate(transpose_list, headers, floatfmt=".2f"))
        punti = getattr(POINTS, color_used[i])
        # cv_plot_display = np.load(path_files/'cv_plot_display.npy')
        for j in range(len(atlas.labels_index)):
            if j in indici:
                coord = np.where(atlas.segmentation_data == atlas.labels_index[j][0])
                atlas.cv_plot_display[coord[0],coord[1],coord[2],:] = atlas.labels_color[j]
        # Plot
        fig_color_probe, ax_color_probe = plt.subplots(1, 1) # to plot the region interested with colors
        IndexTracker_pi_col(ax_color_probe, atlas.cv_plot_display/255, atlas.Edges, atlas.pixdim, P[i].Plane, P[i].Slice, punti[0], punti[1], line_fit)
        ax_color_probe.set_title('Probe %d\n(%s)' % (i+1, color_used[i]))
        plt.show()
        
        # Write and save txt file with probe info
        pn = "Probe_%s.txt" % color_used[i]
        f = open(os.path.join(path_info, pn),"w+")
        f.write('Analyze probe: \n\n ')
        f.write('Probe length: %.2f mm \n\n' % Length[0])
        f.write('Estimated probe insertion angle:  \n')
        f.write('%.2f degrees in the anterior direction \n' % deg_ant)
        f.write('%.2f degrees in the lateral direction\n\n' % deg_lat)
        f.write(tabulate(transpose_list, headers, floatfmt=".2f"))
        f.close()
    
    ax1.axis(xmin=0,xmax=100*n+20)
    ax1.axis(ymin=0,ymax=max_val)
    ax1.axis('off')
    
    
    
    
    # plot all the probes together
    p3d = vedo.Plotter(title='Brain viewer', size=(700, 700), pos=(250, 0))
    if n == 1:
         p3d.show(mesh, getattr(pr,color_used[0]), getattr(L, color_used[0]), __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 2:
         p3d.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(L, color_used[0]), getattr(L, color_used[1]), __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 3:
         p3d.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(L, color_used[0]),getattr(L, color_used[1]), getattr(L, color_used[2]), __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 4:
         p3d.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 5:
         p3d.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), getattr(L, color_used[4]),  __doc__,
         axes=0, viewup="z", bg='black',
         )
    elif n == 6:
         p3d.show(mesh, getattr(pr,color_used[0]), getattr(pr,color_used[1]), getattr(pr,color_used[2]), getattr(pr,color_used[3]), getattr(pr,color_used[4]), getattr(pr,color_used[5]), getattr(L, color_used[0]), getattr(L, color_used[1]), getattr(L, color_used[2]), getattr(L, color_used[3]), getattr(L, color_used[4]), getattr(L, color_used[5]), __doc__,
         axes=0, viewup="z", bg='black',
         )
         
atlas = AtlasLoader(atlas_folder='/Users/jingyig/Work/Kavli/PyCode/vitlab/racer/waxholm_atlas', atlas_version='v3')
probe_folder = '/Users/jingyig/Work/Kavli/PyCode/vitlab/racer/waxholm_atlas/probes2'
vis3d_probe_track(atlas, probe_folder)