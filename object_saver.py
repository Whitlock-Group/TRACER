#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:36:26 2020

@author: jacopop
"""

# Store the transformed image features        
class save_transform(object):    
    def __init__(self, tracker, coord, image, nolines):
        self.Slice = tracker
        self.Transform_points = coord
        self.Transform = image
        self.Transform_withoulines = nolines  
        
# Store probe features                
class save_probe(object):
    def __init__(self, a, b, plane, probe_counter):
        self.Slice = a
        self.Probe = b    
        self.Plane = plane
        self.Counter = probe_counter
        
class save_probe_insertion(object):
    def __init__(self, coord, plane, probe_counter):
        self.Probe = coord
        self.Plane = plane
        self.Counter = probe_counter        
        
        
# object for the clicked probes
class probe_obj(object):
    pass