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
    lib_path = os.path.join(root_dir, 'lib')
    
    vis = 0
    is_train = 0
    
    model = 'CPN152' # option 'CPN18', 'CPN50', 'CPN101', 'CPN152'
    channel_settings = [[512, 256, 128, 64], [2048, 1024, 512, 256]] #'CPN18'
    
    num_class = 80
    img_path = os.path.join(root_dir, '../data/351/test_data')
    csv_file_path = os.path.join(root_dir, "../data/351/test.csv")

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    
    #data_shape = (192, 256) #h w
    #output_shape = (48, 64)
    
    data_shape = (256, 512) #h w
    output_shape = (64, 128)
    
    featuremap_scale = 4

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(cfg.lib_path)
#add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))