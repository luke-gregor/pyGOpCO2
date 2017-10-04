from . import diagnostic_plots
from . import level0_to_level1
from . import level1_to_level2

lvl1 = level0_to_level1
lvl2 = level1_to_level2
dp = diagnostic_plots

import numpy
numpy.seterr(over='ignore')