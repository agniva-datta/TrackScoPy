# -*- coding: utf-8 -*-
"""

- This is a python file containing all default parameters for swim-mode detection

- Author: Agniva Datta @ University of Potsdam, Germany

"""

# Parameters specific to  experiment (free to adapt):

# For bacteria

ARROW_LENGTH = 45   # Radius of search during arrow-plots in pixels
ARROW_LIFETIME = 500   # Delay of search during arrow-plots in frames
BAC_SIZE = 8    # Minimum size of bacteria (cell body+flagella) in pixels
BACKGROUND_FACTOR = 0.015   # Percentage of image to be eliminated as background/100

default_parameters_swim_mode_detection = [ARROW_LENGTH, 
                                          ARROW_LIFETIME, 
                                          BAC_SIZE, 
                                          BACKGROUND_FACTOR]