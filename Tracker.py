#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:31:04 2020

@author: jacopop
"""

# Class to scroll the atlas slices (coronal, sagittal and horizontal)    
class IndexTracker(object):
    def __init__(self, ax, X, pixdim, p):
        self.ax = ax
        self.plane = p.lower()
        ax.set_title('Atlas viewer')
        print('\nuse scroll wheel to navigate the atlas \n')

        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols = X.shape
            self.ind = 653
            self.im = ax.imshow(self.X[:, self.ind, :].T, origin="lower", extent=[0, 512*pixdim, 0, 512*pixdim], cmap='gray')
        elif self.plane == 's':
            self.slices, rows, cols = X.shape
            self.ind = 246                
            self.im = ax.imshow(self.X[self.ind, :, :].T, origin="lower", extent=[0 ,1024*pixdim, 0, 512*pixdim], cmap='gray')            
        elif self.plane == 'h':  
            rows, cols, self.slices = X.shape
            self.ind = 440       
            self.im = ax.imshow(self.X[:, :, self.ind].T, origin="lower", extent=[0, 512*pixdim, 0, 1024*pixdim], cmap='gray')            
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.plane == 'c':
            self.im.set_data(self.X[:, self.ind, :].T)
        elif self.plane == 's':
            self.im.set_data(self.X[self.ind, :, :].T)
        elif self.plane == 'h':
            self.im.set_data(self.X[:, :, self.ind].T)
        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()                      
        
# Class to scroll the overlayed atlas slices (coronal, sagittal and horizontal)            
class IndexTracker_g(object):
    def __init__(self, ax, X, pixdim, p, S):
        self.ax = ax
        self.plane = p.lower()
        ax.set_title('Atlas viewer')
        print('\nuse scroll wheel to navigate the atlas \n')
        
        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[:, self.ind, :], origin="lower", alpha=0.5, extent=[0, 512*pixdim, 512*pixdim, 0], cmap='gray')
        elif self.plane == 's':
            self.slices, rows, cols = X.shape
            self.ind = S                
            self.im = ax.imshow(self.X[self.ind, :, :].T, origin="lower", extent=[0 ,1024*pixdim, 512*pixdim , 0], cmap='gray')            
        elif self.plane == 'h':  
            rows, cols, self.slices = X.shape
            self.ind = S       
            self.im = ax.imshow(self.X[:, :, self.ind].T, origin="lower", extent=[0, 512*pixdim, 1024*pixdim, 0], cmap='gray')  
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()        
        
    def update(self):
        if self.plane == 'c':
            self.im.set_data(self.X[:, self.ind, :])
        elif self.plane == 's':
            self.im.set_data(self.X[self.ind, :, :])
        elif self.plane == 'h':
            self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()      

# Class to scroll the atlas slices with probe (coronal, sagittal and horizontal)    
class IndexTracker_p(object):
    def __init__(self, ax, X, pixdim, p, S):
        self.ax = ax
        self.plane = p.lower()
        ax.set_title('Atlas viewer')
        print('\nuse scroll wheel to navigate the atlas \n')

        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[:, self.ind, :].T, origin="lower", extent=[0, 512*pixdim, 0, 512*pixdim], cmap='gray')
        elif self.plane == 's':
            self.slices, rows, cols = X.shape
            self.ind = S                
            self.im = ax.imshow(self.X[self.ind, :, :].T, origin="lower", extent=[0 ,1024*pixdim, 0, 512*pixdim], cmap='gray')            
        elif self.plane == 'h':  
            rows, cols, self.slices = X.shape
            self.ind = S       
            self.im = ax.imshow(self.X[:, :, self.ind].T, origin="lower", extent=[0, 512*pixdim, 0, 1024*pixdim], cmap='gray')            
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.plane == 'c':
            self.im.set_data(self.X[:, self.ind, :].T)
        elif self.plane == 's':
            self.im.set_data(self.X[self.ind, :, :].T)
        elif self.plane == 'h':
            self.im.set_data(self.X[:, :, self.ind].T)
        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()        
        
        
# Class to scroll the boundaries atlas slices (coronal, sagittal and horizontal)            
class IndexTracker_b(object):
    def __init__(self, ax, X, pixdim, p, S):
        self.ax = ax
        self.plane = p.lower()
        ax.set_title('Atlas viewer')
        print('\nuse scroll wheel to navigate the atlas \n')
        
        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[:, self.ind, :], origin="lower", alpha=0.5, extent=[0, 512*pixdim, 0, 512*pixdim,], cmap='gray')
        elif self.plane == 's':
            self.slices, rows, cols = X.shape
            self.ind = S                
            self.im = ax.imshow(self.X[:, :, self.ind].T, origin="lower", alpha=0.5, extent=[0 ,1024*pixdim, 0, 512*pixdim], cmap='gray')            
        elif self.plane == 'h':  
            rows, cols, self.slices = X.shape
            self.ind = S       
            self.im = ax.imshow(self.X[self.ind, :, :].T, origin="lower", alpha=0.5, extent=[0, 512*pixdim, 0, 1024*pixdim], cmap='gray')  
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()        
        
    def update(self):
        if self.plane == 'c':
            self.im.set_data(self.X[:, self.ind, :])
        elif self.plane == 's':
            self.im.set_data(self.X[:, :, self.ind])
        elif self.plane == 'h':
            self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()           
        

# Class to scroll the color atlas slices (coronal, sagittal and horizontal)            
class IndexTracker_c(object):
    def __init__(self, ax, X, pixdim, p, S):
        self.ax = ax
        self.plane = p.lower()
        ax.set_title('Atlas viewer')
        print('\nuse scroll wheel to navigate the atlas \n')
        
        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols, color = X.shape
            self.ind = S
            self.L = self.X.transpose((2,1,0,3))
            self.im = ax.imshow(self.L[:, self.ind, :], origin="lower", alpha=0.5, extent=[0, 512*pixdim, 0, 512*pixdim ], cmap='gray')
        elif self.plane == 's':
            self.slices, rows, cols, color= X.shape
            self.ind = S
            self.L = self.X.transpose((0,2,1,3))            
            self.im = ax.imshow(self.L[self.ind, :, :], origin="lower", alpha=0.5, extent=[0, 1024*pixdim, 0, 512*pixdim], cmap='gray')            
        elif self.plane == 'h':  
            rows, cols, self.slices, color = X.shape
            self.ind = S
            self.L = self.X.transpose((1,0,2,3))
            self.im = ax.imshow(self.L[:, :, self.ind], origin="lower", alpha=0.5, extent=[0, 512*pixdim,  0, 1024*pixdim ], cmap='gray')  
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()        
        
    def update(self):
        if self.plane == 'c':
            self.im.set_data(self.L[:, self.ind, :])
        elif self.plane == 's':
            self.im.set_data(self.L[self.ind, :, :])
        elif self.plane == 'h':
            self.im.set_data(self.L[:, :, self.ind])
        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()                
                        
                       
import numpy as np    
import matplotlib.pyplot as plt  
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
# Class to scroll the atlas slices with probe (coronal, sagittal and horizontal)    
class IndexTracker_pi(object):

    def __init__(self, ax, X, pixdim, p, S, unique_slice, probe_x, probe_y, probe_colors, probe_selecter, line_fit):
        
        self.ax = ax
        self.plane = p.lower()
        self.unique_slice = unique_slice
        self.probe_x = probe_x
        self.probe_y = probe_y
        self.probe_colors = probe_colors
        self.probe_selecter = probe_selecter
        self.line_fit = line_fit                
        ax.set_title('Atlas viewer')
        print('\nuse scroll wheel to navigate the atlas \n')

        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[:, self.ind, :].T, origin="lower", extent=[0, 512*pixdim, 0, 512*pixdim], cmap='gray')
        elif self.plane == 's':
            self.slices, rows, cols = X.shape
            self.ind = S                
            self.im = ax.imshow(self.X[self.ind, :, :].T, origin="lower", extent=[0 ,1024*pixdim, 0, 512*pixdim], cmap='gray')            
        elif self.plane == 'h':  
            rows, cols, self.slices = X.shape
            self.ind = S       
            self.im = ax.imshow(self.X[:, :, self.ind].T, origin="lower", extent=[0, 512*pixdim, 0, 1024*pixdim], cmap='gray') 
        self.points = [plt.scatter(self.probe_x[i], self.probe_y[i], color=self.probe_colors[self.probe_selecter], s=2) for i in range(len(self.probe_x))]
        # plot the probe
        if self.line_fit.direction[0] == 0:
            self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);           
        else:
            self.m, self.b =  np.polyfit(self.probe_x, self.probe_y, 1)
            self.line = plt.plot(np.array(sorted(self.probe_x)), self.m*np.array(sorted(self.probe_x)) + self.b,color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);           
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
            #self.points.remove()
            self.line.pop(0).remove()
            if self.ind not in self.unique_slice:
                self.points = [plt.scatter(self.probe_x[i], self.probe_y[i], color=lighten_color(self.probe_colors[self.probe_selecter], 0.4), s=2) for i in range(len(self.probe_x))]
                if self.line_fit.direction[0] == 0:
                    self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),color=lighten_color(self.probe_colors[self.probe_selecter], 0.4), linestyle='dashed', linewidth=0.8)
                else:
                    self.line = plt.plot(np.array(sorted(self.probe_x)), self.m*np.array(sorted(self.probe_x)) + self.b,color=lighten_color(self.probe_colors[self.probe_selecter], 0.4), linestyle='dashed', linewidth=0.8)
            else:
                self.points = [plt.scatter(self.probe_x[i], self.probe_y[i], color=self.probe_colors[self.probe_selecter], s=2) for i in range(len(self.probe_x))]
                if self.line_fit.direction[0] == 0:
                    self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);    
                else:
                    # plot the probe
                    self.line = plt.plot(np.array(sorted(self.probe_x)), self.m*np.array(sorted(self.probe_x)) + self.b,color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);    
        else:
            self.ind = (self.ind - 1) % self.slices
            #self.points.remove()
            self.line.pop(0).remove()
            if self.ind not in self.unique_slice:                
                self.points = [plt.scatter(self.probe_x[i], self.probe_y[i], color=lighten_color(self.probe_colors[self.probe_selecter], 0.4), s=2) for i in range(len(self.probe_x))]
                if self.line_fit.direction[0] == 0:
                    self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),color=lighten_color(self.probe_colors[self.probe_selecter], 0.4), linestyle='dashed', linewidth=0.8)
                else:
                    self.line = plt.plot(np.array(sorted(self.probe_x)), self.m*np.array(sorted(self.probe_x)) + self.b,color=lighten_color(self.probe_colors[self.probe_selecter], 0.4), linestyle='dashed', linewidth=0.8)
            else:
                self.points = [plt.scatter(self.probe_x[i], self.probe_y[i], color=self.probe_colors[self.probe_selecter], s=2) for i in range(len(self.probe_x))]
                # plot the probe
                if self.line_fit.direction[0] == 0:
                    self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)),color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);    
                else:
                    self.line = plt.plot(np.array(sorted(self.probe_x)), self.m*np.array(sorted(self.probe_x)) + self.b,color=self.probe_colors[self.probe_selecter], linestyle='dashed', linewidth=0.8);    
        self.update()

    def update(self):
        if self.plane == 'c':
            self.im.set_data(self.X[:, self.ind, :].T)
        elif self.plane == 's':
            self.im.set_data(self.X[self.ind, :, :].T)
        elif self.plane == 'h':
            self.im.set_data(self.X[:, :, self.ind].T)
        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()        


# Class to scroll the atlas slices with probe (coronal, sagittal and horizontal) and color of selected regions   
class IndexTracker_pi_col(object):

    def __init__(self, ax, X, edges, pixdim, p, S, unique_slice, probe_x, probe_y, line_fit):
        self.ax = ax
        self.plane = p.lower()
        self.unique_slice = unique_slice
        self.probe_x = probe_x
        self.probe_y = probe_y
        self.line_fit = line_fit
        
        ax.set_title('Atlas viewer')

        self.X = X
        self.edges =edges
        if self.plane == 'c':
            rows, self.slices, cols, color = X.shape
            self.ind = S
            self.L = self.X.transpose((2,1,0,3))
            self.im = ax.imshow(self.L[:, self.ind, :], origin="lower", extent=[0, 512*pixdim, 0, 512*pixdim])                       
            self.im2 = ax.imshow(self.edges[:, self.ind, :], origin="lower", alpha=0.5, extent=[0, 512*pixdim, 0, 512*pixdim,], cmap='gray')
        elif self.plane == 's':
            self.slices, rows, cols, color= X.shape
            self.ind = S
            self.L = self.X.transpose((0,2,1,3))                  
            self.im = ax.imshow(self.L[self.ind, :, :], origin="lower", extent=[0 ,1024*pixdim, 0, 512*pixdim])            
            self.im2 = ax.imshow(self.edges[self.ind, :, :], origin="lower", alpha=0.5, extent=[0, 1024*pixdim, 0, 512*pixdim,], cmap='gray')
        elif self.plane == 'h':  
            rows, cols, self.slices, color = X.shape
            self.ind = S
            self.L = self.X.transpose((1,0,2,3)) 
            self.im = ax.imshow(self.L[:, :, self.ind], origin="lower", extent=[0, 512*pixdim, 0, 1024*pixdim]) 
            self.im2 = ax.imshow(self.edges[:, :, self.ind], origin="lower", alpha=0.5, extent=[0, 512*pixdim, 0, 1024*pixdim,], cmap='gray')
        # plot the probe
        if self.line_fit.direction[0] == 0:
            self.line = plt.plot(np.array(sorted(self.probe_x)), np.array(sorted(self.probe_y)), linestyle='dashed', linewidth=0.8);        
        else:
            self.m, self.b =  np.polyfit(self.probe_x, self.probe_y, 1)
            self.line = plt.plot(np.array(sorted(self.probe_x)), self.m*np.array(sorted(self.probe_x)) + self.b,color='black', linestyle='dashed', linewidth=0.8);        
        self.ax.set_ylabel('slice %d' % self.ind)
        self.im.axes.figure.canvas.draw()                  
        
        