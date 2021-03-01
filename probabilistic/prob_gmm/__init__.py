import os
import sys

curr_dir = os.path.basename(os.path.abspath(os.curdir))
# See __init__.py in folder "toy_example" for an explanation.
if curr_dir == 'prob_gmm' and '../..' not in sys.path:
    sys.path.insert(0, '../..')

# Initialize plotting environment.
from utils import misc
misc.configure_matplotlib_params()
