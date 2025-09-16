# -*- coding: utf-8 -*-
"""

- This is a python file containing all default parameters for image analysis

- Author: Agniva Datta @ University of Potsdam, Germany

"""

import numpy as np

# Parameters specific to  experiment (free to adapt):

# For bacteria

framerate = 100 # frames per second
magnification =  60 # magnification in image
px_size_camera = 6.5 # micrometers per px 
max_particle_speed = 150 # maximum speed of particle in micrometers/second
memory_b = 1 # for missing particle in frames (can be 0, 1, 2, 5 or 10)
dim = 2 # dimension of image: 2 for 2D tracking
threshold_factor = 1 # threshold factor to modify segmentation (in the range 0.95 to 1.05: less than 1 if no detections, more than 1 if noisy detections)
blur_factor = 5 # kernel size to blur image (in the range 2 to 30, increase the value for weak signals)

SMOOTH_TRACKS_LIM = 5   # maximum  duration of a track  to consider for smoothing
SMOOTH_TRACKS_SPREAD = 25  # threshold spread of pixels above which tracks are considered for smoothing

# Parameters specific to analysis (not recommended to alter):

nbins = 1024
R = np.round(0.3*magnification/px_size_camera)


default_parameters_image_analysis = [framerate, 
                                     magnification, 
                                     px_size_camera, 
                                     max_particle_speed, 
                                     memory_b, dim, 
                                     threshold_factor, 
                                     blur_factor,
                                     SMOOTH_TRACKS_LIM, 
                                     SMOOTH_TRACKS_SPREAD, 
                                     nbins, 
                                     R]
