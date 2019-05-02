DATADIR='./data'
FIGUREDIR='./figure'
import os
import sys
from scipy.io import loadmat
from spectral_density import *
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt



def real_data_analsis(file_name):

    file = os.path.join(DATADIR, file_name)
    mat = loadmat(file)
    '''
    print(type(mat))
    print(mat.keys())
    print(mat['vol'].shape)
    '''



    ts = mat['vol'].T
    model_info = {}
    model_info['model'] = 'ma'
    model_info['weights'] = None
    model_info['span'] = 14
    model_info['stdev'] = 1

    spec_est = SpecEst(ts, model_info, individual_level=True, simu=False)
    heat_maps = spec_est.query_heat_map()
    ave_sm = heat_maps['sm']['ave']
    print(heat_maps['sm']['ave'])
    threshold_matrices = []
    for threshold in  np.arange(0, np.max(abs(ave_sm)), np.max(abs(ave_sm))/20):
        print(threshold)
        value = np.copy(ave_sm)
        value[abs(value)<threshold] = 0
        threshold_matrices.append(value)
    return threshold_matrices



def show_threshold_effects():
    pass


if __name__ == "__main__":
    file_name = 'NIH-101_20100329_ts.mat'
    threshold_matrices = real_data_analsis('NIH-101_20100329_ts.mat')
    for i, spd in enumerate(threshold_matrices):
        plt.figure()
        heatmap(spd)
        #plt.pause(5)
        graph_name = file_name+'-'+str(i)+'.pdf'
        plt.savefig(graph_name)
        plt.close('all')





