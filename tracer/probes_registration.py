#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:18:44 2021

@author: admin
"""


from __future__ import print_function
# Import libraries

import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math 
import mplcursors
from skimage import io, transform
import pickle 
from six.moves import input 
from skspatial.objects import Line


from .index_tracker import IndexTracker, IndexTracker_g, IndexTracker_p
# create objects to svae transformations and probes
from .ObjSave import save_transform, probe_obj, save_probe



class ProbesRegistration(object):
    """
    Purpose
    -------------
    For register probes.

    Inputs
    -------------
    atlas :
    processed_histology_folder :
    show_hist :
    probe_name :
    

    Outputs
    -------------
    A list contains ...

    """

    def __init__(self, atlas, processed_histology_folder=None, show_hist=False, probe_name='probe'):
        
        self.atlas = atlas
        self.processed_histology_folder = processed_histology_folder
        self.show_hist = show_hist
        self.probe_name = probe_name

        # Lists for the points clicked in atlas and histology
        self.coords_atlas = []
        self.coords_hist = []
        self.coords_probe_temp_w = []
        self.coords_probe_temp_g = []
        self.coords_probe_temp_p = []
        self.coords_probe_temp_b = []
        self.coords_probe_temp_y = []
        self.coords_probe_temp_o = []
        self.coords_probe_temp_r = []
        # Object for clicked probes
        self.coords_probe = probe_obj()
        # Lists for the points plotted in atlas and histology
        self.redp_atlas = []
        self.redp_hist = []
        self.rednum_atlas = []
        self.rednum_hist = []
        # List of probe points
        self.p_probe_trans = []
        self.p_probe_grid = []
        # Initialize probe counter and selecter
        self.probe_counter = 0
        self.probe_selecter = 0
        self.flag = 0

        # probes have different colors
        self.probe_colors = ['purple', 'blue', 'yellow', 'orange', 'red', 'green']
        
        if processed_histology_folder is not None:
            if not os.path.exists(self.processed_histology_folder):
                raise Exception('Please give the correct folder.')
    
            # The Transformed images will be saved in a subfolder of process histology called transformations
            self.path_transformed = os.path.join(self.processed_histology_folder, 'transformations')
            if not os.path.exists(self.path_transformed):
                os.mkdir(self.path_transformed)
    
            self.path_probes = os.path.join(self.processed_histology_folder, 'probes')
            if not os.path.exists(self.path_probes):
                os.mkdir(self.path_probes)
    
            self.jj = 0
            self.img_hist_temp = []
            self.img_hist = []
            self.names = []
            for fname in os.listdir(self.processed_histology_folder):
                image_path = os.path.join(self.processed_histology_folder, fname)
                if image_path[-4:] not in ['.jpg', 'jpeg']:
                    continue
                self.img_hist_temp.append(Image.open(image_path).copy())
                self.img_hist.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
                self.names.append(fname)
        
        
        # Insert the plane of interest
        self.plane = str(input('Select the plane coronal (c), sagittal (s), or horizontal (h): ')).lower()
        # Check if the input is correct
        while self.plane != 'c' and self.plane != 's' and self.plane != 'h':
            print('Error: Wrong plane name \n')
            self.plane = str(input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')).lower()
        
        
        # Display the ATLAS
        # resolution
        self.dpi_atl = 25.4 / self.atlas.pixdim
        
        # Bregma coordinates
        self.textstr = 'Bregma (mm): c = %.3f, h = %.3f, s = %.3f \nBregma (voxels): c = 623, h = 440, s = 246' % (
            653 * self.atlas.pixdim, 440 * self.atlas.pixdim, 246 * self.atlas.pixdim)
        # these are matplotlib.patch.Patch properties
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Figure
        self.fig, self.ax = plt.subplots(1, 1)  # , figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl), dpi=dpi_atl)
        # scroll cursor
        self.tracker = IndexTracker(self.ax, self.atlas.atlas_data, self.atlas.pixdim, self.plane)
        self.fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
        # place a text box with bregma coordinates in bottom left in axes coords
        self.ax.text(0.03, 0.03, self.textstr, transform=self.ax.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
        
        if self.plane == 'c':
            # dimensions
            self.d1 = 512
            self.d2 = 512
            d3 = 1024
        elif self.plane == 's':
            # dimensions
            self.d1 = 1024
            self.d2 = 512
            d3 = 512
        elif self.plane == 'h':
            # dimensions
            self.d1 = 512
            self.d2 = 1024
            d3 = 512
        self.ax.format_coord = self.format_coord
        plt.show()
        # Fix size and location of the figure window
        self.mngr = plt.get_current_fig_manager()
        self.mngr.window.setGeometry(800, 300, self.d1, self.d2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        print('\nControls: \n')
        print('--------------------------- \n')
        print('g: activate gridlines \n')
        print('u: load saved transform and atlas location \n')
        print('t: activate mode where clicks are logged for transform \n')
        print('d: delete most recent transform point \n')
        print('h: transform histology to adapt to the atlas \n')
        print("b: simple overlay to scroll through brain regions \n")
        print('p: overlay to the atlas \n')
        print('x: save transform and current atlas location \n')
        print('v: activate color atlas mode \n\n')
        print('r: activate mode where clicks are logged for probe \n')
        print('c: delete most recent probe point \n')
        print('n: add a new probe \n')
        print('w: enable probe viewer mode for current probe  \n')
        print('e: save current probe \n')
        print('--------------------------- \n')
        
        if self.processed_histology_folder is not None:
            if self.show_hist:
                self.plot_hist()
                self.fig_hist.canvas.mpl_connect('key_press_event', self.on_key)
            # Set up the figure
            plt.ioff()
            self.fig_trans, self.ax_trans = plt.subplots(1, 1)
            self.mngr_trans = plt.get_current_fig_manager()
            self.mngr_trans.window.setGeometry(200, 350, self.d2, self.d1)
            self.fig_trans.canvas.mpl_connect('key_press_event', self.on_key)

    
    def format_coord(self, x, y):
        # display the coordinates relative to the bregma when hovering with the cursor
        if self.plane == 'c':
            AP = self.tracker.ind * self.atlas.pixdim - 623 * self.atlas.pixdim
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
            
        
    def plot_hist(self):
        # get the pixel dimension
        self.dpi_hist = self.img_hist_temp[self.jj].info['dpi'][1]
        self.pixdim_hist = 25.4 / self.dpi_hist  # 1 inch = 25,4 mm
        # Show the HISTOLOGY
        # Set up figure
        self.fig_hist, self.ax_hist = plt.subplots(1, 1, figsize=(
        float(self.img_hist[self.jj].shape[1]) / self.dpi_hist, float(self.img_hist[self.jj].shape[0]) / self.dpi_hist))
        self.ax_hist.set_title("Histology viewer")
        # Show the histology image
        self.ax_hist.imshow(self.img_hist_temp[self.jj],
                       extent=[0, self.img_hist[self.jj].shape[1] * self.pixdim_hist, self.img_hist[self.jj].shape[0] * self.pixdim_hist, 0])
        # Remove axes tick
        plt.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
        # Remove cursor position
        self.ax_hist.format_coord = lambda x, y: ''
        plt.show()
        # Fix size and location of the figure window
        self.mngr_hist = plt.get_current_fig_manager()
        self.mngr_hist.window.setGeometry(150, 300, self.d1, self.d2)
        
        
        
        
        

    
    def onclick_atlas(self, event):
        # Mouse click function to store coordinates. Leave a red dot when a point is clicked, Atlas
        ix, iy = event.xdata / self.atlas.pixdim, event.ydata / self.atlas.pixdim
        self.clicka += 1
        # assign global variable to access outside of function
        self.coords_atlas.append((ix, iy))
        self.redp_atlas.extend(plt.plot(event.xdata, event.ydata, 'ro', markersize=2))
        self.rednum_atlas.append(plt.text(event.xdata, event.ydata, self.clicka, fontsize=8, color='red'))
        self.fig.canvas.draw()
        self.active = 'atlas'
        return

    
    def onclick_hist(self, event):
        # Mouse click function to store coordinates. Leave a red dot when a point is clicked, HISTOLOGY
        xh, yh = event.xdata / self.pixdim_hist, event.ydata / self.pixdim_hist
        self.clickh += 1
        # assign global variable to access outside of function
        self.coords_hist.append((xh, yh))
        self.redp_hist.extend(plt.plot(event.xdata, event.ydata, 'ro', markersize=2))
        self.rednum_hist.append(plt.text(event.xdata, event.ydata, self.clickh, fontsize=8, color='red'))
        self.fig_hist.canvas.draw()
        self.active = 'hist'
        return

    
    def show_annotation(self, sel):
        # Show the names of the regions
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
        # plot  point and register all the clicked points
        px, py = event.xdata / self.atlas.pixdim, event.ydata / self.atlas.pixdim
        # assign global variable to access outsid e of function
        if self.probe_counter == 0:
            self.coords_probe_temp_w.append((px, py))
            self.p_probe_grid.extend(
                self.ax_grid.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            self.p_probe_trans.extend(
                self.ax_trans.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_w)
        elif self.probe_counter == 1:
            self.coords_probe_temp_g.append((px, py))
            self.p_probe_grid.extend(
                self.ax_grid.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            self.p_probe_trans.extend(
                self.ax_trans.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_g)
        elif self.probe_counter == 2:
            self.coords_probe_temp_p.append((px, py))
            self.p_probe_grid.extend(
                self.ax_grid.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            self.p_probe_trans.extend(
                self.ax_trans.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_p)
        elif self.probe_counter == 3:
            self.coords_probe_temp_b.append((px, py))
            self.p_probe_grid.extend(
                self.ax_grid.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            self.p_probe_trans.extend(
                self.ax_trans.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_b)
        elif self.probe_counter == 4:
            self.coords_probe_temp_y.append((px, py))
            self.p_probe_grid.extend(
                self.ax_grid.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            self.p_probe_trans.extend(
                self.ax_trans.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_y)
        elif self.probe_counter == 5:
            self.coords_probe_temp_o.append((px, py))
            self.p_probe_grid.extend(
                self.ax_grid.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            self.p_probe_trans.extend(
                self.ax_trans.plot(event.xdata, event.ydata, color=self.probe_colors[self.probe_counter], marker='o', markersize=1))
            setattr(self.coords_probe, self.probe_colors[self.probe_counter], self.coords_probe_temp_o)
        self.fig_grid.canvas.draw()
        self.fig_trans.canvas.draw()
        return

    def on_key2(self, event):
        if event.key == 'n':
            # add a new probe, the function in defined in onclick_probe
            if self.probe_counter + 1 < len(self.probe_colors):
                self.probe_counter += 1
                print('probe %d added (%s)' % (self.probe_counter, self.probe_colors[self.probe_counter]))
            else:
                print('Cannot add more probes')
                self.probe_counter = len(self.probe_colors)
    
        elif event.key == 'c':
            print('Delete clicked probe point')
            if len(getattr(self.coords_probe, self.probe_colors[0])) != 0:
                if len(getattr(self.coords_probe, self.probe_colors[self.probe_counter])) != 0:
                    getattr(self.coords_probe, self.probe_colors[self.probe_counter]).pop(-1)  # remove the point from the list
                    self.p_probe_trans[-1].remove()  # remove the point from the plot
                    self.fig_trans.canvas.draw()
                    self.p_probe_trans.pop(-1)
                    self.p_probe_grid[-1].remove()  # remove the point from the plot
                    self.fig_grid.canvas.draw()
                    self.p_probe_grid.pop(-1)
                elif len(getattr(self.coords_probe, self.probe_colors[self.probe_counter])) == 0:
                    self.probe_counter -= 1
                    try:
                        getattr(self.coords_probe, self.probe_colors[self.probe_counter]).pop(-1)  # remove the point from the list
                        self.p_probe_trans[-1].remove()  # remove the point from the plot
                        self.fig_trans.canvas.draw()
                        self.p_probe_trans.pop(-1)
                        self.p_probe_grid[-1].remove()  # remove the point from the plot
                        self.fig_grid.canvas.draw()
                        self.p_probe_grid.pop(-1)
                    except:
                        pass
                    
    def re_load_images(self, da_image_name):
        # if the histology and atlas have been already overlayed in a previous study it is possible to load it and keep working from that stage
        # and start recording the probes
        print('\nLoad image and slice')
        fmn = os.path.join(self.path_transformed, da_image_name + '.pkl')
        IM = pickle.load(open(fmn, "rb"))
        self.tracker.ind = IM.Slice
        self.img2 = IM.Transform
        self.img_warped = IM.Transform_withoulines
        # open the wreaped figure
        self.ax_trans.imshow(self.img_warped, origin="lower",
                             extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
        self.ax_trans.set_title("Histology adapted to atlas")
        plt.show()
        # open the overlayed figure
        self.fig_grid, self.ax_grid = plt.subplots(1, 1)
        self.overlay = self.ax_grid.imshow(self.img2, origin="lower",
                                           extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
        self.ax_grid.text(0.15, 0.05, self.textstr, transform=self.ax.transAxes, fontsize=6, verticalalignment='bottom',
                          bbox=self.props)
        self.ax_grid.format_coord = self.format_coord
        self.ax_grid.set_title("Histology and atlas overlayed")
        plt.show()
        self.cursor = mplcursors.cursor(self.overlay, hover=True)
        self.cursor.connect('add', self.show_annotation)
        self.mngr_grid = plt.get_current_fig_manager()
        self.mngr_grid.window.setGeometry(200, 350, self.d2, self.d1)
        self.flag = 1
        print('Image loaded')
    
    # Reaction to key pressed
    def on_key(self, event):
        if event.key == 't':
            print('\nRegister %s' % os.path.splitext(self.names[self.jj])[0])
            print('Select at least 4 points in the same order in both figures')
            # Call click func on atlas
            self.clicka = 0
            self.cid_atlas = self.fig.canvas.mpl_connect('button_press_event', self.onclick_atlas)
            # Call click func on hist
            self.clickh = 0
            self.cid_hist = self.fig_hist.canvas.mpl_connect('button_press_event', self.onclick_hist)
            
        elif event.key == 'h':
            print('Transform histology to adapt to the atlas')
            # get the projective transformation from the set of clicked points
            t = transform.ProjectiveTransform()
            t.estimate(np.float32(self.coords_atlas), np.float32(self.coords_hist))
            img_hist_tempo = np.asanyarray(self.img_hist_temp[self.jj])
            self.img_warped = transform.warp(img_hist_tempo, t, output_shape=(self.d1, self.d2), order=1, clip=False)  # , mode='constant',cval=float('nan'))
            # Show the  transformed figure
            # fig_trans, ax_trans = plt.subplots(1, 1)#, figsize=(float(d1)/dpi_atl,float(d2)/dpi_atl))
            # =============================================================================
            #         # Select all black pixels
            #         black = np.array([0, 0, 0])
            #         mask = np.abs(img_warped - black).sum(axis=2) < 0.03
            #         img_warped[mask] = [1, 1, 1]
            # =============================================================================
            self.ax_trans.imshow(self.img_warped, origin="lower", extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
            self.ax_trans.set_title("Histology adapted to atlas")
            self.fig_trans.canvas.draw()
            plt.show()
        
        elif event.key == 'b':
            print('Simple overlay to scroll through brain regions')
            # SIMPLE OVERLAY
            # here you can scroll the atlas grid
            self.fig_g, self.ax_g = plt.subplots(1, 1)
            self.ax_g.imshow(self.img_warped, origin="lower", extent=[0, self.d1 * self.atlas.pixdim, self.d2 * self.atlas.pixdim, 0])
            self.tracker2 = IndexTracker_g(self.ax_g, self.atlas.Edges, self.atlas.pixdim, self.plane, self.tracker.ind)
            self.fig_g.canvas.mpl_connect('scroll_event', self.tracker2.onscroll)
            # ax_g.format_coord = format_coord
            self.ax_g.set_title("Histology and atlas overlayed")
            plt.show()
            # Remove axes tick
            plt.tick_params(axis='both', which='both', bottom=False, left=False, top=False, labelbottom=False,
                            labelleft=False)
        
        elif event.key == 'p':
            print('Overlay to the atlas')
            # get the edges of the colors defined in the label
            if self.plane == 'c':
                self.edges = cv2.Canny(np.uint8((self.atlas.cv_plot[:, self.tracker.ind, :] * 255).transpose((1, 0, 2))), 100, 200)
            elif self.plane == 's':
                self.edges = cv2.Canny(np.uint8((self.atlas.cv_plot[self.tracker.ind, :, :] * 255).transpose((1, 0, 2))), 100, 200)
            elif self.plane == 'h':
                self.edges = cv2.Canny(np.uint8((self.atlas.cv_plot[:, :, self.tracker.ind] * 255).transpose((1, 0, 2))), 100, 200)

            self.fig_grid, self.ax_grid = plt.subplots(1, 1)
            # position of the lines
            self.CC = np.where(self.edges == 255)
            self.img2 = (self.img_warped).copy()
            # get the lines in the warped figure
            self.img2[self.CC] = 0.5  # here change grid color (0 black, 1 white)
            self.overlay = self.ax_grid.imshow(self.img2, origin="lower", extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
            self.ax_grid.text(0.15, 0.05, self.textstr, transform=self.ax.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
            self.ax_grid.format_coord = self.format_coord
            self.ax_grid.set_title("Histology and atlas overlayed")
            plt.show()
            self.cursor = mplcursors.cursor(self.overlay, hover=True)
            self.cursor.connect('add', self.show_annotation)
            self.mngr_grid = plt.get_current_fig_manager()
            self.mngr_grid.window.setGeometry(850, 350, self.d2, self.d1)
        
        elif event.key == 'd':
            print('Delete clicked point')
            if self.active == 'atlas':
                try:
                    self.coords_atlas.pop(-1)  # remove the point from the list
                    self.rednum_atlas[-1].remove()  # remove the numbers from the plot
                    self.redp_atlas[-1].remove()  # remove the point from the plot
                    self.fig.canvas.draw()
                    self.rednum_atlas.pop(-1)
                    self.redp_atlas.pop(-1)
                    self.clicka -= 1
                except:
                    pass
            else:
                try:
                    self.rednum_hist[-1].remove()  # remove the numbers from the plot
                    self.coords_hist.pop(-1)  # remove the point from the list
                    self.redp_hist[-1].remove()  # remove the point from the plot
                    self.fig_hist.canvas.draw()
                    self.redp_hist.pop(-1)
                    self.rednum_hist.pop(-1)
                    self.clickh -= 1
                except:
                    pass
        elif event.key == 'x':
            print('\nSave image and slice')
            # Create and save slice, clicked points, and image info
            S = save_transform(self.tracker.ind, [self.coords_hist, self.coords_atlas], self.img2, self.img_warped)  # Saving the object
            da_image_name = self.names[self.jj][:-4]
            print(da_image_name)
            fnm = os.path.join(self.path_transformed, da_image_name + '.pkl')
            with open(fnm, 'wb') as f:
                pickle.dump(S, f)
            # Save the images
            fig_name = da_image_name + '_transformed_withoutlines.jpeg'
            self.fig_trans.savefig(os.path.join(self.path_transformed, fig_name))
            print('\nImage saved')
        
        elif event.key == 'v':
            print('Colored Atlas on')
            self.fig_color, self.ax_color = plt.subplots(1, 1)
            self.ax_color.imshow(self.img2, extent=[0, self.d1 * self.atlas.pixdim, 0, self.d2 * self.atlas.pixdim])
            if self.plane == 'c':
                self.ax_color.imshow(self.atlas.cv_plot[:, self.tracker.ind, :].transpose((1, 0, 2)), origin="lower",
                                extent=[0, self.d1 * self.atlas.pixdim, self.d2 * self.atlas.pixdim, 0], alpha=0.5)
            elif self.plane == 's':
                self.ax_color.imshow(self.atlas.cv_plot[self.tracker.ind, :, :].transpose((1, 0, 2)), origin="lower",
                                extent=[0, self.d1 * self.atlas.pixdim, self.d2 * self.atlas.pixdim, 0], alpha=0.5)
            elif self.plane == 'h':
                self.ax_color.imshow(self.atlas.cv_plot[:, :, self.tracker.ind].transpose((1, 0, 2)), origin="lower",
                                extent=[0, self.d1 * self.atlas.pixdim, self.d2 * self.atlas.pixdim, 0], alpha=0.5)
            self.ax_color.set_title("Histology and colored atlas")
            plt.show()
        
        elif event.key == 'r':
            print('Register probe %d' % (self.probe_counter + 1))
            try:
                plt.close(self.fig_g)
            except:
                pass
            try:
                plt.close(self.fig_color)
            except:
                pass

            # Call click func
            self.cid_trans = self.fig_trans.canvas.mpl_connect('button_press_event', self.onclick_probe)
            self.cid_trans2 = self.fig_trans.canvas.mpl_connect('key_press_event', self.on_key2)
            
        elif event.key == 'w':
            try:
                print('probe %d view mode' % (self.probe_selecter + 1))
                self.L = getattr(self.coords_probe, self.probe_colors[self.probe_selecter])
                self.probe_x = []
                self.probe_y = []
                for i in range(len(self.L)):
                    self.probe_x.append(self.L[i][0] * self.atlas.pixdim)
                    self.probe_y.append(self.L[i][1] * self.atlas.pixdim)
                self.fig_probe, self.ax_probe = plt.subplots(1, 1)
                self.trackerp = IndexTracker_p(self.ax_probe, self.atlas.atlas_data, self.atlas.pixdim, self.plane, self.tracker.ind)
                self.fig_probe.canvas.mpl_connect('scroll_event', self.trackerp.onscroll)
                self.ax_probe.text(0.15, 0.05, self.textstr, transform=self.ax_probe.transAxes, fontsize=6, verticalalignment='bottom', bbox=self.props)
                self.ax_probe.format_coord = self.format_coord
                self.ax_probe.set_title("Probe viewer")
                plt.show()
                self.cursor = mplcursors.cursor(self.fig_probe, hover=True)
                self.cursor.connect('add', self.show_annotation)
                self.mngr_probe = plt.get_current_fig_manager()
                self.mngr_probe.window.setGeometry(900, 400, self.d2, self.d1)
                # plot the clicked points
                plt.scatter(self.probe_x, self.probe_y, color=self.probe_colors[self.probe_selecter], s=2)  # , marker='o', markersize=1)
                # plot the probe
                self.pts = np.array((self.probe_x, self.probe_y)).T
                # fit the probe
                self.line_fit = Line.best_fit(self.pts)
                if self.line_fit.direction[0] == 0:
                    plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)), color=self.probe_colors[self.probe_selecter],
                             linestyle='dashed', linewidth=0.8)
                else:
                    m, b = np.polyfit(self.probe_x, self.probe_y, 1)
                    plt.plot(np.array(sorted(self.probe_x)), m * np.array(sorted(self.probe_x)) + b,
                             color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8)
                self.probe_selecter += 1
            except:
                print('No more probes to visualize')
                pass
        
        elif event.key == 'e':
            self.fig_trans.canvas.mpl_disconnect(self.cid_trans)
            self.fig_trans.canvas.mpl_disconnect(self.cid_trans2)
            # When saving probes use names in increasing order (Alphabetical or numerical) from the one with the first clicked point to the one with the last clicked point.
            # Since the order of the clicked points determin the starting and ending of the probe
            
            # Create and save slice, clicked probes
            P = save_probe(self.tracker.ind, self.coords_probe, self.plane, self.probe_counter)  # Saving the object
            # MAC
            da_probe_name = self.probe_name + str(self.probe_counter) + '.pkl'
            fnm = os.path.join(self.path_probes, da_probe_name)
            with open(fnm, 'wb') as F:
                pickle.dump(P, F)  # Create and save slice, clicked points, and image info
            print('Probe points saved')
            
            try:
                # Close figures and clear variables
                plt.close(self.fig_grid)
                try:
                    plt.close(self.fig_probe)
                except:
                    pass
                for i in range(len(self.coords_atlas)):
                    self.coords_atlas.pop(-1)  # remove the point from the list
                    self.rednum_atlas[-1].remove()  # remove the numbers from the plot
                    self.redp_atlas[-1].remove()  # remove the point from the plot
                    self.fig.canvas.draw()
                    self.redp_atlas.pop(-1)
                    self.rednum_atlas.pop(-1)
                    self.rednum_hist[-1].remove()  # remove the numbers from the plot
                    self.coords_hist.pop(-1)  # remove the point from the list
                    self.redp_hist[-1].remove()  # remove the point from the plot
                    self.fig_hist.canvas.draw()
                    self.redp_hist.pop(-1)
                    self.rednum_hist.pop(-1)
                
                for j in range(len(self.probe_colors)):
                    try:
                        for i in range(len(getattr(self.coords_probe, self.probe_colors[j]))):
                            getattr(self.coords_probe, self.probe_colors[j]).pop(-1)  # remove the point from the list
                            self.p_probe_trans[-1].remove()  # remove the point from the plot
                            self.fig_trans.canvas.draw()
                            self.p_probe_trans.pop(-1)
                            self.p_probe_grid.pop(-1)
                    except:
                        pass
                self.clicka = 0
                self.clickh = 0
                
                self.probe_selecter = 0
                # Disconnnect the registartion of the clicked points to avoid double events
                self.jj += 1
                self.fig.canvas.mpl_disconnect(self.cid_atlas)
                self.fig_hist.canvas.mpl_disconnect(self.cid_hist)
                # =============================================================================
                #             fig_trans.canvas.mpl_disconnect(cid_trans)
                #             #fig_trans.canvas.mpl_disconnect(cid_trans2)
                # =============================================================================

                # OPEN A NEW HISTOLOGY FOR NEXT REGISTRATION
                # get the pixel dimension
                self.dpi_hist = self.img_hist_temp[self.jj].info['dpi'][1]
                self.pixdim_hist = 25.4 / self.dpi_hist  # 1 inch = 25,4 mm
                # Show the HISTOLOGY
                # Set up figure
                # fig_hist, ax_hist = plt.subplots(1, 1, figsize=(float(img_hist[jj].shape[1])/dpi_hist,float(img_hist[jj].shape[0])/dpi_hist))
                self.ax_hist.set_title("Histology viewer")
                # Show the histology image
                self.ax_hist.imshow(self.img_hist_temp[self.jj],
                               extent=[0, self.img_hist[self.jj].shape[1] * self.pixdim_hist, self.img_hist[self.jj].shape[0] * self.pixdim_hist, 0])
                # Remove axes tick
                plt.tick_params(
                    axis='both',
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False)  # labels along the bottom edge are off
                # Remove cursor position
                self.ax_hist.format_coord = lambda x, y: ''
                self.fig_hist.canvas.draw()
                plt.show()
                # Fix size and location of the figure window
                self.mngr_hist = plt.get_current_fig_manager()
                self.mngr_hist.window.setGeometry(150, 300, self.d1, self.d2)
            except:
                print('\nNo more histology slice to register')
                plt.close('all')
    
    



