import os
import sys

curr_dir = os.path.basename(os.path.abspath(os.curdir))
# See __init__.py in folder "toy_example" for an explanation.
trgt_dir = os.path.abspath(os.path.join('..', '..'))
if curr_dir == 'regression' and trgt_dir != sys.path[0]:
    sys.path.insert(0, trgt_dir)

# Initialize plotting environment.
from utils import misc
misc.configure_matplotlib_params()
