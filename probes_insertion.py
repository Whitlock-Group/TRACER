"""
@ author    : Jacopo Paglia,   2021-03
@ modified  : Jingyi Fuglstad, 2021-07
@ maintainer: Jingyi Fuglstad, jingyi.guo@ntnu.no
              Pearl Saldanha,  pearl.saldanha@ntnu.no
@ copyright : whitlock group @ KISN @ NTNU
"""

# Import libraries
import math 
import os
import os.path
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from tabulate import tabulate
import mplcursors
from scipy.spatial import distance
from six.moves import input

# fit the probe
from skspatial.objects import Line


from ObjSave import probe_obj, save_probe_insertion
from Tracker import IndexTracker, IndexTracker_g, IndexTracker_pi, IndexTracker_b, IndexTracker_c, IndexTracker_pi_col
from Atlas_Loader import AtlasLoader


class ProbesInsertion(object):
    """
    Purpose
    -------------
    To insert probes, pre experiment.

    Inputs
    -------------
    atlas :
    probe_folder :
    
    """

    def __init__(self, atlas, probe_folder):
        
        self.atlas = atlas
        self.probe_folder = probe_folder
        
        
        if not os.path.exists(self.probe_folder):
            raise Exception('Please give the correct folder.')
        
        self.probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']
        
        # PROBE
        self.max_probe_length = 10  # maximum length of probe shank is 10mm
        self.probe_widht = 0.07
        self.probe_thickness = 0.024
        self.probe_tip_length = 0.175
        self.total_electrodes = 960  # total number of recording sites
        self.electrode = 0.012  # Electrode size is 12x12 micron
        self.vert_el_dist = 0.02
        # There are 2 electrodes every 0.02 mm

        # Lists for the points clicked in atlas and histology
        self.coords_atlas = []
        self.coords_probe_temp_w = []
        self.coords_probe_temp_g = []
        self.coords_probe_temp_p = []
        self.coords_probe_temp_b = []
        self.coords_probe_temp_y = []
        self.coords_probe_temp_o = []
        self.coords_probe_temp_r = []
        # Object for clicked probes
        self.coords_probe = probe_obj()
        # List of probe points
        self.p_probe = []
        # Initialize probe counter and selecter
        self.probe_counter = 0
        self.probe_selecter = 0
        self.probe_selecter_u = 0

        self.Pp = []

        self.flag_color = 0
        self.flag_boundaries = 0
        self.flag_names = 0
        self.flag = 0
        
        
        self.plane = str(input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')).lower()
        # Check if the input is correct
        while self.plane != 'c' and self.plane != 's' and self.plane != 'h':
            print('Error: Wrong plane name \n')
            self.plane = str(input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')).lower()
            
        
        print('\nControls: \n')
        print('--------------------------- \n')
        print('scroll: move between slices \n')
        print('g: add/remove gridlines \n')
        print('b: add/remove name of current region \n')
        print('o: add/remove viewing boundaries \n')
        print('v: add/remove atlas color \n')
        print('r: toggle mode where clicks are logged for probe \n')
        print('n: trace a new probe \n')
        print('e: save probes \n')
        print('w: enable/disable probe viewer mode for current probe  \n')
        print('c: delete most recent probe point \n')
        print('--------------------------- \n')
        
        
        
        # Display the ATLAS
        # resolution
        self.dpi_atl = 25.4 / self.atlas.pixdim
        # Bregma coordinates
        self.textstr = 'Bregma (mm): c = %.3f, h = %.3f, s = %.3f \nBregma (voxels): c = 653, h = 440, s = 246' % (653 * self.atlas.pixdim, 440 * self.atlas.pixdim, 246 * self.atlas.pixdim)
        # these are matplotlib.patch.Patch properties
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Figure
        self.fig, self.ax = plt.subplots(1, 1)  #, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl), dpi=dpi_atl)
        # scroll cursor
        self.tracker = IndexTracker(self.ax, self.atlas.atlas_data, self.atlas.pixdim, self.plane)
        self.fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
        # place a text box with bregma coordinates in bottom left in axes coords
        self.ax.text(0.03, 0.03, self.textstr, transform=self.ax.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
        if self.plane == 'c':
            # dimensions
            self.d1 = 512
            self.d2 = 512
            self.ax.format_coord = self.format_coord
        elif self.plane == 's':
            # dimensions
            self.d2 = 1024
            self.d1 = 512
            self.ax.format_coord = self.format_coord
        elif self.plane == 'h':
            # dimensions
            self.d2 = 512
            self.d1 = 1024
            self.ax.format_coord = self.format_coord
        plt.show()
        # Fix size and location of the figure window
        self.mngr = plt.get_current_fig_manager()
        self.mngr.window.setGeometry(600, 200, self.d2 * 2, self.d1 * 2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)


    def re_load_probes(self, probe_name):
        print('\nLoad probe')
        c_file = open(os.path.join(self.probe_folder, probe_name + '.pkl'), "rb")
        tdata = pickle.load(c_file)
        self.Pp.append(tdata)
        c_file.close()
        self.flag = 1
        print('Probe loaded')

    
    def format_coord(self, x, y):
        # display the coordinates relative to the bregma when hovering with the cursor
        if self.plane == 'c':
            AP = self.tracker.ind * self.atlas.pixdim - 653 * self.atlas.pixdim
            ML = x - 246 * self.atlas.pixdim
            Z = y - 440 * self.atlas.pixdim
            if ML > 0:
                return 'AP=%1.4f, ML=R%1.4f, z=%1.4f' % (AP, abs(ML), Z)
            else:
                return 'AP=%1.4f, ML=L%1.4f, z=%1.4f' % (AP, abs(ML), Z)
        elif self.plane == 's':
            AP = x - 653 * self.atlas.pixdim
            ML = self.tracker.ind * self.atlas.pixdim - 246 * self.atlas.pixdim
            Z = y - 440 * self.atlas.pixdim
            if ML > 0:
                return 'AP=%1.4f, ML=R%1.4f, z=%1.4f' % (AP, abs(ML), Z)
            else:
                return 'AP=%1.4f, ML=L%1.4f, z=%1.4f' % (AP, abs(ML), Z)
        elif self.plane == 'h':
            AP = y - 653 * self.atlas.pixdim
            ML = x - 246 * self.atlas.pixdim
            Z = self.tracker.ind * self.atlas.pixdim - 440 * self.atlas.pixdim
            if ML > 0:
                return 'AP=%1.4f, ML=R%1.4f, z=%1.4f' % (AP, abs(ML), Z)
            else:
                return 'AP=%1.4f, ML=L%1.4f, z=%1.4f' % (AP, abs(ML), Z)

    def show_annotation(self, sel):
        if self.flag_names == 1:
            sel.annotation.set_visible(True)
        elif self.flag_names == 0:
            sel.annotation.set_visible(False)
        xi, yi = sel.target / self.atlas.pixdim
        if self.plane == 'c':
            if np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(xi)[1]), self.tracker.ind, int(math.modf(yi)[1])], axis=1)).size:
                Text = self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[
                    int(math.modf(xi)[1]), self.tracker.ind, int(math.modf(yi)[1])], axis=1))[0, 0]]
            else:
                # display nothing
                Text = ' '
        elif self.plane == 's':
            if np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[self.tracker.ind, int(math.modf(xi)[1]), int(math.modf(yi)[1])], axis=1)).size:
                Text = self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[
                    self.tracker.ind, int(math.modf(xi)[1]), int(math.modf(yi)[1])], axis=1))[0, 0]]
            else:
                # display nothing
                Text = ' '
        elif self.plane == 'h':
            if np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(xi)[1]), int(math.modf(yi)[1]), self.tracker.ind], axis=1)).size:
                Text = self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[
                    int(math.modf(xi)[1]), int(math.modf(yi)[1]), self.tracker.ind], axis=1))[0, 0]]
            else:
                # display nothing
                Text = ' '
        sel.annotation.set_text(Text)

    
    def onclick_probe(self, event):
        px, py = event.xdata, event.ydata
        # assign global variable to access outside of function
        if self.probe_counter == 0:
            self.coords_probe_temp_w.append((px, py, self.tracker.ind))
            self.p_probe.extend(
                self.ax.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=2))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_w)
        elif self.probe_counter == 1:
            self.coords_probe_temp_g.append((px, py, self.tracker.ind))
            self.p_probe.extend(
                self.ax.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=2))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_g)
        elif self.probe_counter == 2:
            self.coords_probe_temp_p.append((px, py, self.tracker.ind))
            self.p_probe.extend(
                self.ax.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=2))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_p)
        elif self.probe_counter == 3:
            self.coords_probe_temp_b.append((px, py, self.tracker.ind))
            self.p_probe.extend(
                self.ax.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=2))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_b)
        elif self.probe_counter == 4:
            self.coords_probe_temp_y.append((px, py, self.tracker.ind))
            self.p_probe.extend(
                self.ax.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=2))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_y)
        elif self.probe_counter == 5:
            self.coords_probe_temp_o.append((px, py, self.tracker.ind))
            self.p_probe.extend(
                self.ax.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=2))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_o)
        self.fig.canvas.draw()
        return

    def on_key2(self, event):
        if event.key == 'n':
            # add a new probe
            if self.probe_counter + 1 < len(self.probe_colors):
                self.probe_counter += 1
                print('probe %d added (%s)' % (self.probe_counter + 1, self.probe_colors[self.probe_counter]))
            else:
                print('Cannot add more probes')
                self.probe_counter = len(self.probe_colors)
    
        elif event.key == 'c':
            print('Delete clicked probe point')
            if len(getattr(self.coords_probe, self.probe_colors[0])) != 0:
                if len(getattr(self.coords_probe, self.probe_colors[self.probe_counter])) != 0:
                    getattr(self.coords_probe, self.probe_colors[self.probe_counter]).pop(-1)  # remove the point from the list
                    self.p_probe[-1].remove()  # remove the point from the plot
                    self.fig.canvas.draw()
                    self.p_probe.pop(-1)
                elif len(getattr(self.coords_probe, self.probe_colors[self.probe_counter])) == 0:
                    self.probe_counter -= 1
                    try:
                        getattr(self.coords_probe, self.probe_colors[self.probe_counter]).pop(-1)  # remove the point from the list
                        self.p_probe[-1].remove()  # remove the point from the plot
                        self.fig.canvas.draw()
                        self.p_probe.pop(-1)
                
                    except:
                        pass
    
    def on_key(self, event):
        if event.key == 'o':
            if self.flag_boundaries == 0:
                print('View boundaries on')
                self.tracker2 = IndexTracker_b(self.ax, self.atlas.Edges, self.atlas.pixdim, self.plane, self.tracker.ind)
                self.fig.canvas.mpl_connect('scroll_event', self.tracker2.onscroll)
                plt.show()
                self.flag_boundaries = 1
            elif self.flag_boundaries == 1:
                print('View boundaries off')
                self.fig.delaxes(self.ax)
                self.ax.clear()
                plt.draw()
                self.fig.add_axes(self.ax)
                plt.draw()
                self.tracker = IndexTracker(self.ax, self.atlas.atlas_data, self.atlas.pixdim, self.plane)
                self.fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
                plt.show()
                self.flag_boundaries = 0
        elif event.key == 'v':
            if self.flag_color == 0:
                print('View colors on')
                self.tracker3 = IndexTracker_c(self.ax, self.atlas.cv_plot, self.atlas.pixdim, self.plane, self.tracker.ind)
                self.fig.canvas.mpl_connect('scroll_event', self.tracker3.onscroll)
                plt.show()
                self.flag_color = 1
            elif self.flag_color == 1:
                print('View colors off')
                self.fig.delaxes(self.ax)
                self.ax.clear()
                plt.draw()
                self.fig.add_axes(self.ax)
                plt.draw()
                self.tracker = IndexTracker(self.ax, self.atlas.atlas_data, self.atlas.pixdim, self.plane)
                self.fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
                plt.show()
                self.flag_color = 0
        elif event.key == 'b':
            # Show the names of the regions
            self.cursor = mplcursors.cursor(hover=True)
            self.cursor.connect('add', self.show_annotation)
            if self.flag_names == 0:
                print("Show region's name on")
                self.flag_names = 1
            elif self.flag_names == 1:
                print("Show region's name off")
                self.flag_names = 0
        elif event.key == 'r':
            print('Register probe %d' % self.probe_counter)
            # Call click func
            self.fig.canvas.mpl_connect('button_press_event', self.onclick_probe)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key2)
        elif event.key == 'e':
            print('\n Save probe')
            # Create and save slice, clicked probes
            print(self.coords_probe)
            print(self.probe_counter)
            P = save_probe_insertion(self.coords_probe, self.plane, self.probe_counter)        # Saving the object
            probe_name = 'Probe%d.pkl' % self.probe_counter
            file_name = os.path.join(self.probe_folder, probe_name)
            a_file = open(file_name, "wb")
            pickle.dump(P, a_file)
            a_file.close()

            print('Probe saved')
            
            
        elif event.key == 'w':
            # if the probe if uploaded from a file
            if self.flag == 1:
                # If I have several probes
                for j in range(len(self.probe_colors)):
                    for k in range(len(self.Pp)):
                        try:
                            PC = getattr(self.Pp[k].Probe, self.probe_colors[j])
                            p_x = []
                            p_y = []
                            probe_slice = []
                            for i in range(len(PC)):
                                p_x.append(PC[i][0])
                                p_y.append(PC[i][1])
                                probe_slice.append(PC[i][2])
                            unique_slice = list(OrderedDict.fromkeys(probe_slice))
                            # get the probe coordinates and the region's names
                            probe_x = []
                            probe_y = []
                            probe_z = []
                            if self.Pp[k].Plane == 'c':
                                for i in range(len(PC)):
                                    probe_x.append(PC[i][0])
                                    probe_y.append(PC[i][2] * self.atlas.pixdim)
                                    probe_z.append(PC[i][1])
                            elif self.Pp[k].Plane == 's':
                                for i in range(len(PC)):
                                    probe_x.append(PC[i][2] * self.atlas.pixdim)
                                    probe_y.append(PC[i][0])
                                    probe_z.append(PC[i][1])
                            elif self.Pp[k].Plane == 'h':
                                for i in range(len(PC)):
                                    probe_x.append(PC[i][0])
                                    probe_y.append(PC[i][1])
                                    probe_z.append(PC[i][2] * self.atlas.pixdim)
                            pts = np.array((probe_x, probe_y, probe_z)).T
                            line_fit = Line.best_fit(pts)
                            # display the probe in a separate window
                            self.fig_probe, self.ax_probe = plt.subplots(1, 1)
                            self.trackerp = IndexTracker_pi(self.ax_probe, self.atlas.atlas_data, self.atlas.pixdim, self.Pp[k].Plane, probe_slice[0], unique_slice, p_x, p_y, self.probe_colors, self.probe_selecter_u, line_fit)
                            self.fig_probe.canvas.mpl_connect('scroll_event', self.trackerp.onscroll)
                            self.ax_probe.text(0.05, 0.95, self.textstr, transform=self.ax_probe.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
                            self.ax_probe.format_coord = self.format_coord
                            self.ax_probe.set_title("Probe %d viewer" % (self.probe_selecter_u + 1))
                            plt.show()
                            self.mngr_probe = plt.get_current_fig_manager()
                            self.mngr_probe.window.setGeometry(650, 250, self.d2 * 2, self.d1 * 2)
                            # get the probe coordinates and the region's names
                            
                            # if no inclination in z direction
                            if line_fit.direction[2] == 0:
                                # if there is NOT inclination in the x direction
                                if line_fit.direction[0] == 0:
                                    # line equations, to derive the send point of the line (aka probe)
                                    z2 = pts[0,2]
                                    x2 = pts[-1,0]
                                    y2 = pts[0,1]
                                    deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                                    deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                                    # position_at_bregma_depth
                                    z0 = 440 * self.atlas.pixdim # correspond at the position of the bregma DV=0
                                    x0 = pts[0,0]
                                    y0 = pts[-1,1]
                                    ML_position = (x0 - 246 * self.atlas.pixdim)
                                    AP_position = (y0 - 653 * self.atlas.pixdim)
                                    X0 = np.array([x0, y0, z0])
                                    X2 = np.array([x2, y2, z2])
                                    # start point for visualization (the first clicked point)
                                    z1 = z2
                                    x1 = pts[0,0]
                                    y1 = pts[0,1]
                                    X1 = np.array([x1, y1, z1])
                                    # end point minus tip length
                                    d = (self.probe_tip_length)
                                    xt = x2
                                    yt = y2-d
                                    zt = z2
                                    Xt = np.array([xt, yt, zt])
                                    # get lenght of the probe
                                    dist = np.linalg.norm(X0 - X2)
                                    dist_check = np.linalg.norm(X0 - Xt)
                                    # check kthat the new end point is before the end of the tip and not after
                                    if dist_check > dist:
                                        xt = x2
                                        yt = y2+d
                                        zt = z2
                                        Xt = np.array([xt, yt, zt])
                                    regions = []
                                    point_along_line = []
                                    s = int(math.modf(X1[1] / self.atlas.pixdim)[1])  # starting point
                                    f = int(math.modf(Xt[1] / self.atlas.pixdim)[1])  # ending point
                                    for y in range(min(s,f), max(s,f)):
                                        x = pts[0,0] / self.atlas.pixdim
                                        z = pts[0,2] / self.atlas.pixdim
                                        if int(math.modf(x)[1]) > 512 or int(math.modf(y)[1]) > 1024 or int(math.modf(z)[1]) > 512:
                                            regions.append('Clear Label')
                                        else:
                                            regions.append(self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                                        point_along_line.append([x,y,z])
                                # if there is inclination in the x direction
                                else:
                                    # line equations, to derive the send point of the line (aka probe)
                                    z2 = pts[0,2]
                                    x2 = pts[-1,0]
                                    y2 = line_fit.point[1] + ((x2 - line_fit.point[0]) / line_fit.direction[0])*line_fit.direction[1]
                                    deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                                    deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                                    # position_at_bregma_depth
                                    z0 = 440 * self.atlas.pixdim  # correspond at the position of the bregma DV=0
                                    x0 = pts[0,0]
                                    y0 = line_fit.point[1]+((x0-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                                    ML_position = (x0-246 * self.atlas.pixdim)
                                    AP_position = (y0-653 * self.atlas.pixdim)
                                    X0 = np.array([x0,y0,z0])
                                    X2 = np.array([x2,y2,z2])
                                    # start point for visualization (the first clicked point)
                                    z1 = z2
                                    x1 = pts[0,0]
                                    y1 = line_fit.point[1]+((x1-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                                    X1 = np.array([x1,y1,z1])
                                    # end point minus tip length
                                    dq = (self.probe_tip_length)**2
                                    div = 1 + (line_fit.direction[1]/line_fit.direction[0])**2
                                    xt = x2 + math.sqrt(dq/div)
                                    yt = line_fit.point[1]+((xt-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                                    zt = z2
                                    Xt = np.array([xt,yt,zt])
                                    # get lenght of the probe
                                    dist = np.linalg.norm(X0-X2)
                                    dist_check = np.linalg.norm(X0-Xt)
                                    # check kthat the new end point is before the end of the tip and not after
                                    if dist_check > dist:
                                        xt = x2 - math.sqrt(dq/div)
                                        yt = line_fit.point[1]+((xt-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                                        zt = z2
                                        Xt = np.array([xt,yt,zt])
                                    regions = []
                                    point_along_line = []
                                    s = int(math.modf(X1[0] / self.atlas.pixdim)[1])  # starting point
                                    f = int(math.modf(Xt[0] / self.atlas.pixdim)[1])  # ending point
                                    for x in range(min(s,f), max(s,f)):
                                        y = line_fit.point[1]/self.atlas.pixdim+((x-line_fit.point[0]/self.atlas.pixdim)/line_fit.direction[0])*line_fit.direction[1]
                                        z = pts[0,2] / self.atlas.pixdim
                                        if int(math.modf(x)[1]) > 512 or int(math.modf(y)[1]) > 1024 or int(math.modf(z)[1]) > 512:
                                            regions.append('Clear Label')
                                        else:
                                            regions.append(self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                                        point_along_line.append([x,y,z])
                            else:
                                # line equations, to derive the end point of the line (aka probe)
                                # the last of the clicked points represent the end point of the line
                                z2 = pts[-1,2]
                                x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                                y2 = line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                                deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                                deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                                # position_at_bregma_depth
                                z0 = 440 * self.atlas.pixdim  # correspond at the position of the bregma DV=0
                                x0 = line_fit.point[0]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                                y0 = line_fit.point[1]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                                ML_position = (x0 - 246 * self.atlas.pixdim)
                                AP_position = (y0 - 653 * self.atlas.pixdim)
                                X0 = np.array([x0,y0,z0])
                                X2 = np.array([x2,y2,z2])
                                # start point for visualization (the first clicked point)
                                z1 = pts[0,2]
                                x1 = line_fit.point[0]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                                y1 = line_fit.point[1]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                                X1 = np.array([x1,y1,z1])
                                # end point minus tip length
                                dq = (self.probe_tip_length)**2
                                div = 1 + (line_fit.direction[0]/line_fit.direction[2])**2 + (line_fit.direction[1]/line_fit.direction[2])**2
                                zt = z2 + math.sqrt(dq/div)
                                xt = line_fit.point[0]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                                yt = line_fit.point[1]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                                Xt = np.array([xt,yt,zt])
                                # get lenght of the probe
                                dist = np.linalg.norm(X0-X2)
                                dist_check = np.linalg.norm(X0-Xt)
                                # check kthat the new end point is before the end of the tip and not after
                                if dist_check > dist:
                                    zt = z2 - math.sqrt(dq/div)
                                    xt = line_fit.point[0]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                                    yt = line_fit.point[1]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                                    Xt = np.array([xt,yt,zt])
                                regions = []
                                point_along_line = []
                                s = int(math.modf(X1[2] / self.atlas.pixdim)[1])  # starting point
                                f = int(math.modf(Xt[2] / self.atlas.pixdim)[1])  # ending point
                                for z in range(min(s,f),max(s,f)):
                                    x = line_fit.point[0]/self.atlas.pixdim+((z-line_fit.point[2]/self.atlas.pixdim)/line_fit.direction[2])*line_fit.direction[0]
                                    y = line_fit.point[1]/self.atlas.pixdim+((z-line_fit.point[2]/self.atlas.pixdim)/line_fit.direction[2])*line_fit.direction[1]
                                    regions.append(self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                                    point_along_line.append([x,y,z])
                            # avoid repetions and reverse the order
                            regioni = list(OrderedDict.fromkeys(regions))[::-1]
                            if 'Clear Label' in regioni:
                                regioni.remove('Clear Label')
                            num_el = []
                            indici = []
                            for re in regioni:
                                # store the index o the region to print only the color of the regions of interest
                                indici.append(self.atlas.labels_name.index(re))
                                # in the case in dont exit and then enter again the region
                                position = [i for i,x in enumerate(regions) if x == re]
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
                            # print insertion coordinates
                            print('\n---Estimated probe insertion---')
                            if ML_position > 0:
                                testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = R%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' %(AP_position, abs(ML_position), dist, deg_ant, deg_lat)
                                print('Entry position at DV = 0: AP = %.2f mm, ML = R%.2f mm' % (AP_position, abs(ML_position)))
                            else:
                                testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = L%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' %(AP_position, abs(ML_position), dist, deg_ant, deg_lat)
                                print('Entry position at DV = 0: AP = %.2f mm, ML = L%.2f fmm'
                                      % (AP_position, abs(ML_position)))
                            print('Insertion distance from the above position: %.2f mm' % dist)
                            print('%.2f degrees in the anterior direction' % deg_ant)
                            print('%.2f degrees in the lateral direction\n' % deg_lat)
                            # print regions and channels
                            LL = [regioni, num_el]
                            headers = [' Regions traversed', 'Channels']
                            numpy_array = np.array(LL)
                            transpose_ll = numpy_array.T
                            transpose_list = transpose_ll.tolist()
                            print(tabulate(transpose_list, headers, floatfmt=".2f"))
                            if self.plane == 'c':
                                regioni.insert(0,'            ---Regions traversed---')
                                if len(regioni)>16:
                                    self.ax_probe.text(0.01, 0.26, testo, transform=self.ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')
                                    B = regioni[:len(regioni)//2]
                                    C = regioni[len(regioni)//2:]
                                    self.ax_probe.text(0.41, 0.26, "\n".join(B), transform=self.ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')
                                    self.ax_probe.text(0.76, 0.26, "\n".join(C), transform=self.ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')
                                else:
                                    self.ax_probe.text(0.01, 0.26, testo, transform=self.ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')
                                    self.ax_probe.text(0.51, 0.26, "\n".join(regioni), transform=self.ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')
                            elif self.plane == 's':
                                self.ax_probe.text(0.15, 0.20, testo, transform=self.ax_probe.transAxes, fontsize=11 ,verticalalignment='top', color = 'w')
                                regioni.insert(0,'            ---Regions traversed---')
                                # if there are too many regions to print
                                if len(regioni) > 7:
                                    B = regioni[:len(regioni)//2]
                                    C = regioni[len(regioni)//2:]
                                    self.ax_probe.text(0.5, 0.25, "\n".join(B), transform=self.ax_probe.transAxes, fontsize=9.5 ,verticalalignment='top', color = 'w')
                                    self.ax_probe.text(0.74, 0.25, "\n".join(C), transform=self.ax_probe.transAxes, fontsize=9.5 ,verticalalignment='top', color = 'w')
                                else:
                                    self.ax_probe.text(0.51, 0.25, "\n".join(regioni), transform=self.ax_probe.transAxes, fontsize=11 ,verticalalignment='top', color = 'w')
                            elif self.plane == 'h':
                                
                                regioni.insert(0,'            ---Regions traversed---')
                                # if there are too many regions to print
                                if len(regioni) > 7:
                                    self.ax_probe.text(0.17, 0.22, testo, transform=self.ax_probe.transAxes, fontsize=8 ,verticalalignment='top', color = 'w')
                                    B = regioni[:len(regioni)//2]
                                    C = regioni[len(regioni)//2:]
                                    self.ax_probe.text(0.01, 0.15, "\n".join(B), transform=self.ax_probe.transAxes, fontsize=6.5 ,verticalalignment='top', color = 'w')
                                    self.ax_probe.text(0.49, 0.15, "\n".join(C), transform=self.ax_probe.transAxes, fontsize=6.4 ,verticalalignment='top', color = 'w')
                                else:
                                    self.ax_probe.text(0.17, 0.22, testo, transform=self.ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')
                                    self.ax_probe.text(0.17, 0.13, "\n".join(regioni), transform=self.ax_probe.transAxes, fontsize=9 ,verticalalignment='top', color = 'w')
                            # here I only color the region of interest
                            for i in range(len(self.atlas.labels_index)):
                                if i in indici:
                                    coord = np.where(self.atlas.segmentation_data == self.atlas.labels_index[i][0])
                                    self.atlas.cv_plot_display[coord[0],coord[1],coord[2],:] = self.atlas.labels_color[i]
                            # Plot
                            self.fig_color, self.ax_color = plt.subplots(1, 1)  # to plot the region interested with colors
                            print(self.Pp[k].Plane)
                            IndexTracker_pi_col(self.ax_color, self.atlas.cv_plot_display/255, self.atlas.Edges, self.atlas.pixdim, self.Pp[k].Plane, probe_slice[0], p_x, p_y, line_fit)
                            plt.show()
                            self.mngr_col = plt.get_current_fig_manager()
                            self.mngr_col.window.setGeometry(650, 250, self.d2 * 2, self.d1 * 2)
                            self.probe_selecter_u += 1
                        except:
                            pass
            else:
                try:
                    print('\nProbe %d view mode' % (self.probe_selecter + 1))
                    L = getattr(self.coords_probe, self.probe_colors[self.probe_selecter])
                    p_x = []
                    p_y = []
                    probe_slice = []
                    for i in range(len(L)):
                        p_x.append(L[i][0])
                        p_y.append(L[i][1])
                        probe_slice.append(L[i][2])
                    unique_slice = list(OrderedDict.fromkeys(probe_slice))
                    # get the probe coordinates and the region's names
                    probe_x = []
                    probe_y = []
                    probe_z = []
                    if self.plane == 'c':
                        for i in range(len(L)):
                            probe_x.append(L[i][0])
                            probe_y.append(L[i][2] * self.atlas.pixdim)
                            probe_z.append(L[i][1])
                    elif self.plane == 's':
                        for i in range(len(L)):
                            probe_x.append(L[i][2] * self.atlas.pixdim)
                            probe_y.append(L[i][0])
                            probe_z.append(L[i][1])
                    elif self.plane == 'h':
                        for i in range(len(L)):
                            probe_x.append(L[i][0])
                            probe_y.append(L[i][1])
                            probe_z.append(L[i][2] * self.atlas.pixdim)
                    pts = np.array((probe_x, probe_y, probe_z)).T
                    # fit the probe
                    line_fit = Line.best_fit(pts)
                    # display the probe in a separate window
                    self.fig_probe, self.ax_probe = plt.subplots(1, 1)
                    self.trackerp = IndexTracker_pi(self.ax_probe, self.atlas.atlas_data, self.atlas.pixdim, self.plane, self.tracker.ind, unique_slice, p_x, p_y, self.probe_colors, self.probe_selecter, line_fit)
                    self.fig_probe.canvas.mpl_connect('scroll_event', self.trackerp.onscroll)
                    self.ax_probe.text(0.05, 0.95, self.textstr, transform=self.ax_probe.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
                    self.ax_probe.format_coord = self.format_coord
                    self.ax_probe.set_title("Probe %d viewer" % (self.probe_selecter + 1))
                    plt.show()
                    self.mngr_probe = plt.get_current_fig_manager()
                    self.mngr_probe.window.setGeometry(650, 250, self.d2 * 2, self.d1 * 2)
                    
                    # if no inclination in z direction
                    if line_fit.direction[2] == 0:
                        # if there is NOT inclination in the x direction
                        if line_fit.direction[0] == 0:
                            # line equations, to derive the send point of the line (aka probe)
                            z2 = pts[0,2]
                            x2 = pts[-1,0]
                            y2 = pts[0,1]
                            deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                            deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                            # position_at_bregma_depth
                            z0 = 440 * self.atlas.pixdim  # correspond at the position of the bregma DV=0
                            x0 = pts[0,0]
                            y0 = pts[-1,1]
                            ML_position = (x0 - 246 * self.atlas.pixdim)
                            AP_position = (y0 - 653 * self.atlas.pixdim)
                            X0 = np.array([x0,y0,z0])
                            X2 = np.array([x2,y2,z2])
                            # start point for visualization (the first clicked point)
                            z1 = z2
                            x1 = pts[0,0]
                            y1 = pts[0,1]
                            X1 = np.array([x1,y1,z1])
                            # end point minus tip length
                            d = (self.probe_tip_length)
                            xt = x2
                            yt = y2-d
                            zt = z2
                            Xt = np.array([xt,yt,zt])
                            # get lenght of the probe
                            dist = np.linalg.norm(X0-X2)
                            dist_check = np.linalg.norm(X0-Xt)
                            # check kthat the new end point is before the end of the tip and not after
                            if dist_check > dist:
                                xt = x2
                                yt = y2+d
                                zt = z2
                                Xt = np.array([xt,yt,zt])
                            regions = []
                            point_along_line = []
                            s = int(math.modf(X1[1] / self.atlas.pixdim)[1])  # starting point
                            f = int(math.modf(Xt[1] / self.atlas.pixdim)[1])  # ending point
                            for y in range(min(s,f), max(s,f)):
                                x = pts[0,0] / self.atlas.pixdim
                                z = pts[0,2] / self.atlas.pixdim
                                if int(math.modf(x)[1]) > 512 or int(math.modf(y)[1]) > 1024 or int(math.modf(z)[1]) > 512:
                                    regions.append('Clear Label')
                                else:
                                    regions.append(self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                                point_along_line.append([x,y,z])
                         # if there is inclination in the x direction
                        else:
                            # line equations, to derive the send point of the line (aka probe)
                            z2 = pts[0,2]
                            x2 = pts[-1,0]
                            y2 = line_fit.point[1]+((x2-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                            deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                            deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                            # position_at_bregma_depth
                            z0 = 440 * self.atlas.pixdim # correspond at the position of the bregma DV=0
                            x0 = pts[0,0]
                            y0 = line_fit.point[1]+((x0-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                            ML_position = (x0-246 * self.atlas.pixdim)
                            AP_position = (y0-653 * self.atlas.pixdim)
                            X0 = np.array([x0,y0,z0])
                            X2 = np.array([x2,y2,z2])
                            # start point for visualization (the first clicked point)
                            z1 = z2
                            x1 = pts[0,0]
                            y1 = line_fit.point[1]+((x1-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                            X1 = np.array([x1,y1,z1])
                            # end point minus tip length
                            dq = (self.probe_tip_length)**2
                            div = 1 + (line_fit.direction[1]/line_fit.direction[0])**2
                            xt = x2 + math.sqrt(dq/div)
                            yt = line_fit.point[1]+((xt-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                            zt = z2
                            Xt = np.array([xt,yt,zt])
                            # get lenght of the probe
                            dist = np.linalg.norm(X0-X2)
                            dist_check = np.linalg.norm(X0-Xt)
                            # check kthat the new end point is before the end of the tip and not after
                            if dist_check > dist:
                                xt = x2 - math.sqrt(dq/div)
                                yt = line_fit.point[1]+((xt-line_fit.point[0])/line_fit.direction[0])*line_fit.direction[1]
                                zt = z2
                                Xt = np.array([xt,yt,zt])
                            regions = []
                            point_along_line = []
                            s = int(math.modf(X1[0]/self.atlas.pixdim)[1])  # starting point
                            f = int(math.modf(Xt[0]/self.atlas.pixdim)[1])  # ending point
                            for x in range(min(s,f), max(s,f)):
                                y = line_fit.point[1]/self.atlas.pixdim+((x-line_fit.point[0]/self.atlas.pixdim)/line_fit.direction[0])*line_fit.direction[1]
                                z = pts[0,2] / self.atlas.pixdim
                                if int(math.modf(x)[1]) > 512 or int(math.modf(y)[1]) > 1024 or int(math.modf(z)[1]) > 512:
                                    regions.append('Clear Label')
                                else:
                                    regions.append(self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                                point_along_line.append([x,y,z])
                    else:
                        # line equations, to derive the  point of the line (aka probe)
                        # the last of the clicked points represent the end point of the line
                        z2 = pts[-1,2]
                        x2 = line_fit.point[0]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                        y2 = line_fit.point[1]+((z2-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                        deg_lat = math.degrees(math.atan(line_fit.direction[0]))
                        deg_ant = math.degrees(math.atan(line_fit.direction[1]))
                        # position_at_bregma_depth
                        z0 = 440 * self.atlas.pixdim # correspond at the position of the bregma DV=0
                        x0 = line_fit.point[0]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                        y0 = line_fit.point[1]+((z0-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                        ML_position = (x0 - 246 * self.atlas.pixdim)
                        AP_position = (y0 - 653 * self.atlas.pixdim)
                        X0 = np.array([x0,y0,z0])
                        X2 = np.array([x2,y2,z2])
                        # start point for visualization (the first clicked point)
                        z1 = pts[0,2]
                        x1 = line_fit.point[0]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                        y1 = line_fit.point[1]+((z1-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                        X1 = np.array([x1,y1,z1])
                        # end point minus tip length
                        dq = (self.probe_tip_length)**2
                        div = 1 + (line_fit.direction[0]/line_fit.direction[2])**2 + (line_fit.direction[1]/line_fit.direction[2])**2
                        zt = z2 + math.sqrt(dq/div)
                        xt = line_fit.point[0]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                        yt = line_fit.point[1]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                        Xt = np.array([xt,yt,zt])
                        # get lenght of the probe
                        dist = np.linalg.norm(X0-X2)
                        dist_check = np.linalg.norm(X0-Xt)
                        # check kthat the new end point is before the end of the tip and not after
                        if dist_check > dist:
                            zt = z2 - math.sqrt(dq/div)
                            xt = line_fit.point[0]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[0]
                            yt = line_fit.point[1]+((zt-line_fit.point[2])/line_fit.direction[2])*line_fit.direction[1]
                            Xt = np.array([xt,yt,zt])
                        regions = []
                        point_along_line = []
                        s = int(math.modf(X1[2] / self.atlas.pixdim)[1])  # starting point
                        f = int(math.modf(Xt[2] / self.atlas.pixdim)[1])  # ending point
                        for z in range(min(s,f),max(s,f)):
                            x = line_fit.point[0]/self.atlas.pixdim+((z-line_fit.point[2]/self.atlas.pixdim)/line_fit.direction[2])*line_fit.direction[0]
                            y = line_fit.point[1]/self.atlas.pixdim+((z-line_fit.point[2]/self.atlas.pixdim)/line_fit.direction[2])*line_fit.direction[1]
                            if int(math.modf(x)[1]) > 512 or int(math.modf(y)[1]) > 1024 or int(math.modf(z)[1]) > 512:
                                regions.append('Clear Label')
                            else:
                                regions.append(self.atlas.labels_name[np.argwhere(np.all(self.atlas.labels_index == self.atlas.segmentation_data[int(math.modf(x)[1]),int(math.modf(y)[1]),int(math.modf(z)[1])], axis=1))[0,0]])
                            point_along_line.append([x,y,z])
                    # avoid repetions and reverse the order
                    regioni = list(OrderedDict.fromkeys(regions))[::-1]
                    if 'Clear Label' in regioni:
                        regioni.remove('Clear Label')
                    num_el = []
                    indici = []
                    for re in regioni:
                        # store the index o the region to print only the color of the regions of interest
                        indici.append(self.atlas.labels_name.index(re))
                        # in the case in dont exit and then enter again the region
                        position = [i for i,x in enumerate(regions) if x == re]
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
                    # print insertion coordinates
                    print('\n---Estimated probe insertion---')
                    if ML_position > 0:
                        testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = R%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' % (AP_position, abs(ML_position), dist, deg_ant, deg_lat)
                        print('Entry position at DV = 0: AP = %.2f mm, ML = R%.2f mm' % (AP_position, abs(ML_position)))
                    else:
                        testo = '            ---Estimated probe insertion--- \nEntry position at DV = 0: AP = %.2f mm, ML = L%.2f mm \nInsertion distance from the above position: %.2f mm \n%.2f degrees in the anterior direction \n%.2f degrees in the lateral direction ' % (AP_position, abs(ML_position), dist, deg_ant, deg_lat)
                        print('Entry position at DV = 0: AP = %.2f mm, ML = L%.2f fmm' % (AP_position, abs(ML_position)))
                    print('Insertion distance from the above position: %.2f mm' % dist)
                    print('%.2f degrees in the anterior direction' % deg_ant)
                    print('%.2f degrees in the lateral direction\n' % deg_lat)
                    # print regions and number of channels
                    LL = [regioni, num_el]
                    headers = [' Regions traversed', 'Channels']
                    numpy_array = np.array(LL)
                    transpose_ll = numpy_array.T
                    transpose_list = transpose_ll.tolist()
                    print(tabulate(transpose_list, headers, floatfmt=".2f"))
                    if self.plane == 'c':
                        # list of regions
                        regioni.insert(0,'            ---Regions traversed---')
                        if len(regioni) > 16:
                            self.ax_probe.text(0.01, 0.26, testo, transform=self.ax_probe.transAxes, fontsize=6.5, verticalalignment='top', color='w')
                            B = regioni[:len(regioni)//2]
                            C = regioni[len(regioni)//2:]
                            self.ax_probe.text(0.41, 0.26, "\n".join(B), transform=self.ax_probe.transAxes, fontsize=6.5, verticalalignment='top', color='w')
                            self.ax_probe.text(0.76, 0.26, "\n".join(C), transform=self.ax_probe.transAxes, fontsize=6.5, verticalalignment='top', color='w')
                        else:
                            self.ax_probe.text(0.01, 0.26, testo, transform=self.ax_probe.transAxes, fontsize=9, verticalalignment='top', color='w')
                            self.ax_probe.text(0.51, 0.26, "\n".join(regioni), transform=self.ax_probe.transAxes, fontsize=9, verticalalignment='top', color='w')
                    elif self.plane == 's':
                        self.ax_probe.text(0.15, 0.20, testo, transform=self.ax_probe.transAxes, fontsize=11, verticalalignment='top', color='w')
                        regioni.insert(0,'            ---Regions traversed---')
                        # if there are too many regions to print
                        if len(regioni) > 7:
                            B = regioni[:len(regioni)//2]
                            C = regioni[len(regioni)//2:]
                            self.ax_probe.text(0.5, 0.25, "\n".join(B), transform=self.ax_probe.transAxes, fontsize=9.5, verticalalignment='top', color='w')
                            self.ax_probe.text(0.74, 0.25, "\n".join(C), transform=self.ax_probe.transAxes, fontsize=9.5, verticalalignment='top', color='w')
                        else:
                            self.ax_probe.text(0.51, 0.25, "\n".join(regioni), transform=self.ax_probe.transAxes, fontsize=11, verticalalignment='top', color='w')
                    elif self.plane == 'h':
                        regioni.insert(0,'            ---Regions traversed---')
                        # if there are too many regions to print
                        if len(regioni) > 7:
                            self.ax_probe.text(0.17, 0.22, testo, transform=self.ax_probe.transAxes, fontsize=8, verticalalignment='top', color='w')
                            B = regioni[:len(regioni) // 2]
                            C = regioni[len(regioni) // 2:]
                            self.ax_probe.text(0.01, 0.15, "\n".join(B), transform=self.ax_probe.transAxes, fontsize=6.5, verticalalignment='top', color='w')
                            self.ax_probe.text(0.49, 0.15, "\n".join(C), transform=self.ax_probe.transAxes, fontsize=6.4, verticalalignment='top', color='w')
                        else:
                            self.ax_probe.text(0.17, 0.22, testo, transform=self.ax_probe.transAxes, fontsize=9, verticalalignment='top', color='w')
                            self.ax_probe.text(0.17, 0.13, "\n".join(regioni), transform=self.ax_probe.transAxes, fontsize=9, verticalalignment='top', color='w')
                            
                    # here I only color the region of interest
                    for i in range(len(self.atlas.labels_index)):
                        if i in indici:
                            coord = np.where(self.atlas.segmentation_data == self.atlas.labels_index[i][0])
                            self.atlas.cv_plot_display[coord[0],coord[1],coord[2],:] = self.atlas.labels_color[i]
                    # Plot
                    self.fig_color, self.ax_color = plt.subplots(1, 1)  # to plot the region interested with colors
                    IndexTracker_pi_col(self.ax_color, self.atlas.cv_plot_display / 255, self.atlas.Edges, self.atlas.pixdim, self.plane, self.tracker.ind, p_x, p_y, line_fit)
                    plt.show()
                    self.mngr_col = plt.get_current_fig_manager()
                    self.mngr_col.window.setGeometry(650, 250, self.d2 * 2, self.d1 * 2)
                    self.probe_selecter += 1
                except:
                    print('No more probes to visualize')
                    pass
                
    







