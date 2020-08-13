import os
import numpy as np

def make_dir(folder):
    if not os.path.exists(folder):
        print('Making directory: {}'.format(folder))
        os.makedirs(folder)
    else:
        print('Existing directory: {}'.format(folder))


