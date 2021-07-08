# Import libraries
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

# fit the probe
from skspatial.objects import Line
#from skspatial.plotting import plot_3d

from .ObjSave import probe_obj, save_probe


class vis_inserted_probes(object):
    """
    
    ----------------------------------------------------
    
    Purpose
    -------------
    Visualise inserted probes in 2D and 3D.

    Inputs
    -------------
    atlas :
    probe_folder: (optional), can be set up later through object method set_data_folder

    Outputs
    -------------
    A object contains the probes information.
    
    Usage
    -------------
    vis_obj = vis_inserted_probes(atlas)
    vis_obj.set_data_folder('path_to_the_inserted_probes_data')
    vis_obj.vis2d()
    vis_obj_vis3d()
    
    """

    def __init__(self, atlas, probe_folder=None):
    
        if probe_folder is not None:
            if not os.path.exists(probe_folder):
                raise Exception('Please give the correct folder.')

        self.atlas = atlas
        self.probe_folder = probe_folder
        
        # PROBE
        self.max_probe_length = 10  # maximum length of probe shank is 10mm
        self.probe_widht = 0.07
        self.probe_thickness = 0.024
        self.probe_tip_length = 0.175
        self.total_electrodes = 960  # total number of recording sites
        self.electrode = 0.012  # Electrode size is 12x12 micron
        self.vert_el_dist = 0.02
        # There are 2 electrodes every 0.02 mm
        
    
        # # Probe colors
        self.probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']

        if probe_folder is not None:
            self.set_data_folder(self.probe_folder)
        
        
    def set_data_folder(self, probe_folder):
        if not os.path.exists(probe_folder):
            raise Exception('Please give the correct folder.')

        self.probe_folder = probe_folder
        # get the all the files in the probe folder
        self.files_probe = []
        for fname in os.listdir(self.probe_folder):
            if fname[-4:] != '.pkl':
                continue
            self.files_probe.append(fname)
    
        self.L = probe_obj()
        self.LINE_FIT = probe_obj()
        self.pr = probe_obj()
        self.xyz = probe_obj()
        self.P = []
        self.color_used_t = []
    
        for f in sorted(self.files_probe):
            ff = os.path.join(probe_folder, f)
            a_file = open(ff, 'rb')
            da_data = pickle.load(a_file)
            self.P.append(da_data)
            a_file.close()
    
        # probe_counter = P[0].Counter
    
        # If I have several probes
        for j in range(len(self.probe_colors)):
            # get the probe coordinates and the region's names
            probe_x = []
            probe_y = []
            probe_z = []
            for k in range(len(self.P)):
                try:
                    PC = getattr(self.P[k].Probe, self.probe_colors[j])
                    if self.P[k].Plane == 'c':
                        for i in range(len(PC)):
                            probe_x.append(PC[i][0])
                            probe_y.append(PC[i][2] * self.atlas.pixdim)
                            probe_z.append(PC[i][1])
                    elif self.P[k].Plane == 's':
                        for i in range(len(PC)):
                            probe_x.append(PC[i][2] * self.atlas.pixdim)
                            probe_y.append(PC[i][0])
                            probe_z.append(PC[i][1])
                    elif self.P[k].Plane == 'h':
                        for i in range(len(PC)):
                            probe_x.append(PC[i][0])
                            probe_y.append(PC[i][1])
                            probe_z.append(PC[i][2] * self.atlas.pixdim)
                    self.pts = np.array((probe_x, probe_y, probe_z)).T
                    # fit the probe
                    line_fit = Line.best_fit(self.pts)
                    # if no inclination in z direction
                    if line_fit.direction[2] == 0:
                        # line equations, to derive the starting and end point of the line (aka probe)
                        # it is impartant that the first point clicked is the most external one, and the last correspond to the end of the probe
                        z1 = self.pts[0,2]
                        x1 = self.pts[0,0]
                        y1 = line_fit.point[1]+((x1-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                        z2 = self.pts[0,2]
                        x2 = self.pts[-1,0]
                        y2 = line_fit.point[1]+((x2-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                    else:
                        # line equations, to derive the starting and end point of the line (aka probe)
                        # it is impartant that the first point clicked is the most external one, and the last correspond to the end of the probe
                        z1 = self.pts[0,2]
                        x1 = line_fit.point[0]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                        y1 = line_fit.point[1]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                        z2 = self.pts[-1,2]
                        x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                        y2 = line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                        # end point minus tip length
                        dq = (self.probe_tip_length)**2
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
                    # get the line to plot
                    l = vedo.Line([x1, y1, z1]/self.atlas.pixdim,[x2, y2, z2]/self.atlas.pixdim,c=self.probe_colors[j], lw=2)
                    # clicked points to display
                    pp = vedo.Points(self.pts / self.atlas.pixdim, c=self.probe_colors[j])  # fast
                    setattr(self.xyz, self.probe_colors[j], [[x1, y1, z1], [xt, yt, zt]])
                    setattr(self.pr, self.probe_colors[j], pp)
                    setattr(self.L, self.probe_colors[j], l)
                    setattr(self.LINE_FIT, self.probe_colors[j], line_fit)
                    self.color_used_t.append(self.probe_colors[j])
                except:
                    pass
        #
        # get only the unique color in order
        self.color_used = list(OrderedDict.fromkeys(self.color_used_t))
        self.n = len(self.color_used)
        
    
    def vis2d(self):
        dist = []
        dist_real = []
        # To plot the probe with colors
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        # compute and display the insertion angle for each probe
        for i in range(0, self.n):
            line_fit = getattr(self.LINE_FIT, self.color_used[i])
            deg_lat = math.degrees(math.atan(line_fit.direction[0]))
            deg_ant = math.degrees(math.atan(line_fit.direction[1]))
            print('\n\nAnalyze %s probe: \n ' % self.color_used[i])
        
            # Get the brain regions traversed by the probe
            X1 = getattr(self.xyz, self.color_used[i])[0]
            X2 = getattr(self.xyz, self.color_used[i])[1]
            s = int(math.modf(X1[2]/self.atlas.pixdim)[1])  # starting point
            f = int(math.modf(X2[2]/self.atlas.pixdim)[1])  # ending point
            # get lenght of the probe
            dist.append(np.linalg.norm(f-s))
            regions = []
            colori = []
            initials = []
            index = []
            point_along_line = []
            if line_fit.direction[2] == 0:
                # position_at_bregma_depth
                z0 = 440 * self.atlas.pixdim  # correspond at the position of the bregma DV=0
                x0 = self.pts[0,0]
                y0 = line_fit.point[1]+((x0-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                ML_position = (x0 - 246 * self.atlas.pixdim)
                AP_position = (y0 - 653 * self.atlas.pixdim)
                X0 = np.array([x0,y0,z0])
                dist_real.append(np.linalg.norm(np.array(X0)-np.array(X2)))
                for x in range(min(s,f), max(s,f)):
                    y = line_fit.point[1]/self.atlas.pixdim+((x-line_fit.point[0]/self.atlas.pixdim)/line_fit.direction[0])*line_fit.direction[1]
                    z = self.pts[0,2]
                    point_along_line.append([x,y,z])
                    if int(math.modf(x)[1]) > 512 or int(math.modf(y)[1]) > 1024 or int(math.modf(z)[1]) > 512:
                        regions.append('Clear Label')
                        colori.append('black')
                        initials.append('CL')
                    else:
                        regions.append(self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                        colori.append(self.atlas.labels_color[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                        initials.append(self.atlas.labels_initial[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
            else:
                # position_at_bregma_depth
                z0 = 440 * self.atlas.pixdim  # correspond at the position of the bregma DV=0
                x0 = line_fit.point[0]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                y0 = line_fit.point[1]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                ML_position = (x0 - 246 * self.atlas.pixdim)
                AP_position = (y0 - 653 * self.atlas.pixdim)
                X0 = np.array([x0,y0,z0])
                dist_real.append(np.linalg.norm(np.array(X0)-np.array(X2)))
                for z in range(min(s,f),max(s,f)):
                    x = line_fit.point[0]/self.atlas.pixdim+((z-line_fit.point[2]/self.atlas.pixdim)/line_fit.direction[2])*line_fit.direction[0]
                    y = line_fit.point[1]/self.atlas.pixdim+((z-line_fit.point[2]/self.atlas.pixdim)/line_fit.direction[2])*line_fit.direction[1]
                    point_along_line.append([x,y,z])
                    if int(math.modf(x)[1]) > 512 or int(math.modf(y)[1]) > 1024 or int(math.modf(z)[1]) > 512:
                        regions.append('Clear Label')
                        colori.append('black')
                        initials.append('CL')
                    else:
                        regions.append(self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                        colori.append(self.atlas.labels_color[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                        initials.append(self.atlas.labels_initial[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                    #index.append(segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])])
            print('Insertion distance from the above position: %.2f mm' % dist_real[i])
            if ML_position > 0:
                testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = R%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' %( AP_position, abs(ML_position), dist_real[i], deg_ant, deg_lat)
                print('Entry position at DV = 0: AP = %.2f mm, ML = R%.2f mm' % (AP_position, abs(ML_position)))
            else:
                testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = L%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' %( AP_position, abs(ML_position), dist_real[i], deg_ant, deg_lat)
                print('Entry position at DV = 0: AP = %.2f mm, ML = L%.2f fmm' % (AP_position, abs(ML_position)))
            print('Estimated %s probe insertion angle: ' % self.color_used[i])
            print('%.2f degrees in the anterior direction' % deg_ant)
            print('%.2f degrees in the lateral direction\n' % deg_lat)
            # count the number of elements in each region to
            counter_regions = dict(Counter(regions))
            regioni = list(OrderedDict.fromkeys(regions))
            iniziali = list(OrderedDict.fromkeys(initials))
            if 'Clear Label' in regioni:
                iniziali.pop(regioni.index('Clear Label'))
                regioni.pop(regioni.index('Clear Label'))
            cc = 0
            jj = 0
            num_el = []
            for re in regioni:
                # in the case in dont exit and then enter again the region
                position = [i for i, x in enumerate(regions) if x == re]
                # if there is only one point in the region
                if len(position) == 1:
                    regional_dist = self.atlas.pixdim
                else:
                    # first point along the line in the region
                    start = [element * self.atlas.pixdim for element in point_along_line[position[0]]]
                    # last point along the line in the region
                    end = [element * self.atlas.pixdim for element in point_along_line[position[-1]]]
                    # length of the part of the probe in the region
                    regional_dist = distance.euclidean(start, end)
                # Number of electrodes in the region
                num_el.append(round(regional_dist/self.vert_el_dist)*2)
                #print(re)
                # proportion of the probe in the given region
                dist_prop = dist[i]/len(regioni)
                color_prop = self.atlas.labels_color[np.argwhere(np.array(self.atlas.labels_name) == re)]
                # plot the probe with the colors of the region traversed
                ax1.add_patch(patches.Rectangle((100*i+5, cc), 17, dist_prop, color=color_prop[0][0]/255))
                plt.text(100*i, max(dist)+10, 'Probe %d\n(%s)' % (i+1, self.color_used[i]), color=self.color_used[i], fontsize=9, fontweight='bold')
                if len(iniziali[jj]) > 7:
                    plt.text(100*i-2, cc+2, '%s-\n%s' % (iniziali[jj][0:5], iniziali[jj][6:]), fontsize=5.6)
                else:
                    plt.text(100*i-2, cc+4, '%s' % (iniziali[jj]), fontsize=6)
                plt.text(100*i+28, cc+4, '%d' % (num_el[jj]), fontsize=6.5)
                jj += 1
                cc = dist_prop + cc
            #indici = list(OrderedDict.fromkeys(index))
            LL = [regioni,  iniziali, num_el]
            headers = [' Regions traversed', 'Initials', 'Channels']
            numpy_array = np.array(LL)
            transpose = numpy_array.T
            transpose_list = transpose.tolist()
            transpose_list.reverse()
            print(tabulate(transpose_list, headers, floatfmt=".2f"))
        lims = (0,max(dist))
        plt.ylim(lims)
        plt.xlim((0,100*self.n+20))
        plt.axis('off')
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(600, 200, 600, 500)


    def vis3d(self):
        vedo.settings.embedWindow(backend=False, verbose=True)
        mask_data = self.atlas.mask_data.transpose((2, 1, 0))
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
        
        
        """Build 2 windows that can interact and share functions"""
        plt1 = vedo.Plotter(title='Brain viewer', size=(700,700), pos=(250,0))
        # plot all the probes together
        if self.n == 1:
             plt1.show(mesh, getattr(self.pr, self.color_used[0]), getattr(self.L, self.color_used[0]), __doc__,
             axes=0, viewup="z", bg='black',
             )
        elif self.n == 2:
             plt1.show(mesh, getattr(self.pr, self.color_used[0]), getattr(self.pr, self.color_used[1]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), __doc__,
             axes=0, viewup="z", bg='black',
             )
        elif self.n == 3:
             plt1.show(mesh, getattr(self.pr, self.color_used[0]), getattr(self.pr, self.color_used[1]), getattr(self.pr, self.color_used[2]), getattr(self.L, self.color_used[0]),getattr(self.L, self.color_used[1]), getattr(self.L, self.color_used[2]), __doc__,
             axes=0, viewup="z",
             bg='black',
             )
        elif self.n == 4:
             plt1.show(mesh, getattr(self.pr, self.color_used[0]), getattr(self.pr, self.color_used[1]), getattr(self.pr, self.color_used[2]), getattr(self.pr,self.color_used[3]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), getattr(self.L, self.color_used[2]), getattr(self.L, self.color_used[3]), __doc__,
             axes=0, viewup="z", bg='black',
             )
        elif self.n == 5:
             plt1.show(mesh, getattr(self.pr, self.color_used[0]), getattr(self.pr, self.color_used[1]), getattr(self.pr, self.color_used[2]), getattr(self.pr,self.color_used[3]), getattr(self.pr,self.color_used[4]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), getattr(self.L, self.color_used[2]), getattr(self.L, self.color_used[3]), getattr(self.L, self.color_used[4]),  __doc__,
             axes=0, viewup="z", bg='black',
             )
        elif self.n == 6:
             plt1.show(mesh, getattr(self.pr, self.color_used[0]), getattr(self.pr, self.color_used[1]), getattr(self.pr, self.color_used[2]), getattr(self.pr,self.color_used[3]), getattr(self.pr,self.color_used[4]), getattr(self.pr,self.color_used[5]), getattr(self.L, self.color_used[0]), getattr(self.L, self.color_used[1]), getattr(self.L, self.color_used[2]), getattr(self.L, self.color_used[3]), getattr(self.L, self.color_used[4]), getattr(self.L, self.color_used[5]), __doc__,
             axes=0, viewup="z", bg='black',
             )





