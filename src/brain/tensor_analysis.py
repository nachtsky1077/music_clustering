import os
import sys
sys.path.append(os.getcwd())
from src.math_utilities.util import coherence, fnorm, tensor_diff
from neural_dataloader import MatLoader
from config import config
import numpy as np
import tensorly as tl
from src.math_utilities.mat_svd_analysis import spectral_analysis
from tensorly.decomposition import tucker

def create_spec_tensor(spec_est, num):
    '''
    :spec_est: the estimation results from SpecEst
    :num: number of frequencies used to create the tensor
    return: a tensor in shape 86 * 200 * num
    '''
    tensor = []
    fnorms = []
    ts_spec_modulus = {}
    freq_idx_list = list(spec_est.keys())
    # calculate fnorm of the spectral density estimation
    for freq_idx in freq_idx_list:
        au_spec_modulus[freq_idx] = abs(coherence(spec_est[freq_idx]))
        fnorms.append((freq_idx, fnorm(ts_spec_modulus[freq_idx], fft=False)))
    fnorms.sort(key=lambda item: item[1], reverse=True)
    for i in range(num):
        tensor.append(ts_spec_modulus[fnorms[i][0]])
    tensor = tl.tensor(tensor)
    return tensor

def create_neural_tensor(mat_loader, num):
    '''
    :mat_loader: neural data matrix loader
    :num: number of matrices to load
    '''
    tensor = []
    for i, filename in enumerate(os.listdir(mat_loader.base_path)):
        if i >= num:
            break
        mat_loader.load_mat_data(filename)
        tensor.append(mat_loader.get_mat_data('vol'))
       
    tensor = tl.tensor(tensor)
    return tensor
    

def tucker_decomp(tensor, rank, sketch=False):
    '''
    :return: tucker decomposition, and approx error
    '''
    core, factors = tucker(tensor, ranks=rank)
    # recover tensor
    full_tensor = tl.tucker_to_tensor(core, factors)
    absolute_error = tensor_diff(tensor, full_tensor)
    relative_error = absolute_error / fnorm(tensor, fft=False)
    return core, factors, absolute_error, relative_error


if __name__ == '__main__':
    data_loader = MatLoader(config['DATA_DIR'])
    #data_loader.load_mat_data('NIH-101_20100329_ts.mat')
    #ts = data_loader.get_mat_data('vol')
    #spec_est = spectral_analysis(ts)
    #neural_tensor = create_tensor(spec_est, 10)
    #print(neural_tensor.shape)
    #print(neural_tensor[0].shape)
    num_mat = 90
    tensor = create_neural_tensor(data_loader, num_mat)
    print(tensor.shape)
    core, factors, absolute_error, relative_error = tucker_decomp(tensor, rank=[num_mat, 86//2, 200])
    print(absolute_error, relative_error)

