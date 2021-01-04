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
        print('use scroll wheel to navigate the atlas \n')

        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols = X.shape
            self.ind = 540
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
        print('use scroll wheel to navigate the atlas \n')
        
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
        print('use scroll wheel to navigate the atlas \n')

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
        print('use scroll wheel to navigate the atlas \n')
        
        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[:, self.ind, :], origin="lower", alpha=0.5, extent=[0, 512*pixdim, 0, 512*pixdim,], cmap='gray')
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
        

# Class to scroll the color atlas slices (coronal, sagittal and horizontal)            
class IndexTracker_c(object):
    def __init__(self, ax, X, pixdim, p, S):
        self.ax = ax
        self.plane = p.lower()
        ax.set_title('Atlas viewer')
        print('use scroll wheel to navigate the atlas \n')
        
        self.X = X
        if self.plane == 'c':
            rows, self.slices, cols, color = X.shape
            self.ind = S
            self.im = ax.imshow(self.X[:, self.ind, :], origin="lower", alpha=0.5, extent=[0, 512*pixdim, 0, 512*pixdim], cmap='gray')
        elif self.plane == 's':
            self.slices, rows, cols, color= X.shape
            self.ind = S                
            self.im = ax.imshow(self.X[self.ind, :, :].T, origin="lower", extent=[0 ,1024*pixdim, 512*pixdim , 0], cmap='gray')            
        elif self.plane == 'h':  
            rows, cols, self.slices, color = X.shape
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
                 
        
        
        
        