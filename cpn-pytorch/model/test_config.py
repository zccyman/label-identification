import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')

    model = 'CPN50' # option 'CPN50', 'CPN101'

    vis = 0

    num_class = 20
    img_path = os.path.join(root_dir, 'data', 'test')
    csv_file_path = root_dir + "/data/test.csv"

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
	
    data_shape = (256, 192)
    output_shape = (64, 48)
    featuremap_scale = 4

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))