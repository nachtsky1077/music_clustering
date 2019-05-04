import scipy.io
import traceback
import numpy as np
import os

class MatLoader():

    def __init__(self, base_path):
        self.base_path = base_path
        self.mat = None

    def base_path_is(self, base_path):
        if base_path != self.base_path:
            self.base_path = base_path

    def load_mat_data(self, filename):
        try:
            mat = scipy.io.loadmat(os.path.join(self.base_path,filename))
        except:
            print('[ERR]: error loading mat file.')
            print(traceback.format_exception())
            mat = None
        self.mat = mat

    def get_mat_data(self, field_name):
        if self.mat is None:
            print('[ERR]: no valid mat loaded.')
            return None
        else:
            return self.mat.get(field_name, None)