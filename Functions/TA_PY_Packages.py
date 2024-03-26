### Functions
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.integrate import odeint
 
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', '{:.2f}'.format)
 
### plotting tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

### Matplotlib config
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 12})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import warnings
warnings.filterwarnings("ignore")