"""
``tracer``
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



from .atlas_loader import *
from .preprocess_histology import *
from .ObjSave import *
from .index_tracker import *
from .probes_registration import *
from .probes_insertion import *
from .virus_registration import *
from .vis3d_registered_virus import *
from .vis_inserted_probes import *
from .vis_registered_probes import *
