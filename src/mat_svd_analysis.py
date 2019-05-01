import os
import scipy.io
import traceback
from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from spectral_density import *

class MatLoader():

    def __init__(self, base_path):
        self.base_path = base_path
        self.mat = None

    def base_path_is(self, base_path):
        if base_path != self.base_path:
            self.base_path = base_path

    def load_mat_data(self, filename):
        try:
            mat = scipy.io.loadmat(self.base_path + filename)
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

class SVDAnalyzer():

    def __init__(self, mat=None):
        self.mat = mat

    def mat_is(self, mat):
        self.mat = mat

    def analyze(self):
        U, s, VT = svd(self.mat)
        self.U = U
        self.s = s
        self.VT = VT

    def svd_estimation(self, rank=10):
        if rank > len(self.s):
            print('[ERR]: rank overlimit.')
            return None
        else:
            s_est = np.append(self.s[:rank], [0] * (len(self.s)-rank))
            tail = np.zeros((len(s_est), self.VT.shape[1]-len(s_est)))
            s_est_mat = np.append(np.diag(s_est), 
                                  tail,
                                  axis=1)
            estimation = np.matmul(self.U, s_est_mat)
            estimation = np.matmul(estimation, self.VT)
            return estimation

    def left_singular_vectors(self):
        return self.U
    
    def singular_values(self):
        return self.s

    def right_singular_vectors(self):
        return self.VT

def plot_singular_values(sigma, filenames):
    # plot the singular values
    fig, ax = plt.subplots()
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Values')
    x = range(len(sigma[0]))
    for i in range(len(sigma)):
        s = sigma[i]
        ax.scatter(x=x, y=s, marker='*', label=filenames[i], alpha=0.5)
    ax.legend()
    return fig



mat_base_path = './dataset_mental/FuncTimeSeries 1/FuncTimeSeries/'
#############################################################################################
# main api
#############################################################################################
def svd_estimation_main(mat_base_path, size=10, rank=10, plot_sigma_decay=False):
    '''
    :mat_base_path: the base path for the .mat files
    :size: number of random .mat files selected to analyze
    :rank: the rank used to do svd estimation \hat{X} = svd_estimation(X, rank)
    :plot_sigma_decay: plot the singular values for each .mat matrix
    '''
    matloader = MatLoader(mat_base_path)
    mat_filenames = os.listdir(mat_base_path)
    sa = SVDAnalyzer()
    sigma = []
    filenames = []
    original_vol = []
    estimated_vol = []
    for filename in np.random.choice(mat_filenames, size=size):
        matloader.load_mat_data(filename)
        vol = matloader.get_mat_data('vol')
        sa.mat_is(vol)
        sa.analyze()
        sigma.append(sa.singular_values())
        filenames.append(filename)

        original_vol.append(vol)
        est_vol = sa.svd_estimation(rank=rank)
        estimated_vol.append(est_vol)

    
    # plot singular values decay scatter
    fig = plot_singular_values(sigma, filenames) if plot_sigma_decay else None

    return original_vol, estimated_vol, fig
    

#############################################################################################
# spectral analysis
#############################################################################################
def spectral_analysis(ts, **kwargs):
    '''
    :ts: the multivariate time series data, NxM, where N is the number of time series, M is the
        number of observations
    :kwargs: other parameters
    notes: function used to do ||FFT(\hat{X})-FFT(X)||_F 
    '''
    model_info = {}
    model_info['model'] = kwargs.get('model', 'ma')
    model_info['weights'] = None
    model_info['span'] = kwargs.get('span', 14)
    model_info['stdev'] = kwargs.get('stdev', 1)

    spec_est = SpecEst(ts, model_info, individual_level=True, simu=False)

    return spec_est

def fnorm(original_vol, estimated_vol):
    norm = []
    # calculate frobenius norm of \hat{vol}-vol, where \hat{vol} is a svd rank=N estimation of the original vol matrix
    for vol, est_vol in zip(original_vol, estimated_vol):
        mat = vol - est_vol
        norm.append(np.linalg.norm(mat))
    return norm



mat_base_path = './dataset_mental/FuncTimeSeries 1/FuncTimeSeries/'
ori_vol, est_vol, fig = svd_estimation_main(mat_base_path)
spec_est = spectral_analysis(ori_vol[0])














