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
from sklearn.preprocessing import normalize

def sort_key_fnorm(spec_est_one_freq):
    return fnorm(abs(coherence(spec_est_one_freq)), fft=False)

def sort_key_std(spec_est_one_freq):
    spec_est_vector = abs(coherence(spec_est_one_freq)).ravel()
    spec_est_vector.reshape(1, spec_est_vector.shape[0])
    # normalize
    spec_est_vector = normalize(spec_est_vector)
    return spec_est_vector.std()

def create_spec_tensor(spec_est, num, sort_key_func):
    '''
    :spec_est: the estimation results from SpecEst
    :num: number of frequencies used to create the tensor
    return: a tensor in shape 86 * 200 * num
    '''
    tensor = []
    freq_idx_info = []
    ts_spec_modulus = {}
    freq_idx_list = list(spec_est.keys())
    # calculate fnorm of the spectral density estimation
    for freq_idx in freq_idx_list:
        ts_spec_modulus[freq_idx] = abs(coherence(spec_est[freq_idx]))
        freq_idx_info.append((freq_idx, sort_key_fnorm(spec_est[freq_idx])))
    freq_idx_info.sort(key=lambda item: item[1], reverse=True)
    for i in range(num):
        tensor.append(ts_spec_modulus[freq_idx_info[i][0]])
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
    data_loader.load_mat_data('NIH-101_20100329_ts.mat')
    ts = data_loader.get_mat_data('vol')
    kwargs = {'selected_freq_index': range(101)}
    spec_est = spectral_analysis(ts.T, **kwargs)
    neural_tensor = create_spec_tensor(spec_est, 5, sort_key_std)

    print(neural_tensor.shape)
    print(neural_tensor[0].shape)
    print(neural_tensor)
    #num_mat = 90
    #tensor = create_neural_tensor(data_loader, num_mat)
    #print(tensor.shape)
    #core, factors, absolute_error, relative_error = tucker_decomp(tensor, rank=[num_mat, 86//2, 200])
    #print(absolute_error, relative_error)

