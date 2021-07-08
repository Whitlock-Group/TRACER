"""
``TRACER``
================

Functions present in tracer are listed below.


For Probes
--------------------------

   ...
   ...

For Virus
--------------

   ...

For Visualisations
------------------

   ...


"""


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


from . import tracer
from .tracer.atlas_loader import *
from .tracer.preprocess_histology import *
from .tracer.ObjSave import *
from .tracer.index_tracker import *
from .tracer.probes_registration import *
from .tracer.probes_insertion import *
from .tracer.virus_registration import *
from .tracer.vis3d_registered_virus import *
from .tracer.vis_inserted_probes import *
from .tracer.vis_registered_probes import *
