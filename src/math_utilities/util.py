import numpy as np
from functools import reduce

def autocovariance(x1, x2, k, N=10000):
    temp = np.array([x1[k:N], x2[:N-k]])
    print(temp)
    cov = np.corrcoef(temp)
    return cov[0][1]


def coherence(spd, flag=True):
    if flag:
        D = np.real(np.diag(1.0/np.sqrt(np.diag(spd))))
        res = reduce(np.dot, [D, spd, D])
        res[np.diag_indices(res.shape[0])] = 0
    else:
        res = spd
    return res

def fnorm(mat, fft):
    '''
    :mat: the matrix
    :fft: a flag indicating whether the matrix is in frequency domain or time domain
    '''
    if fft:
        mat_conj = np.conj(mat)
        fnorm = np.trace(np.matmul(mat, mat_conj.T))
    else:
        fnorm = np.linalg.norm(mat)
    return fnorm

def matrix_estimation_error(original_vol, estimated_vol, fft):
    diff = original_vol - estimated_vol
    norm_diff = fnorm(diff, fft)
    norm_ori = fnorm(original_vol, fft)
    return norm_diff, norm_diff / norm_ori

def tensor_diff(tensor1, tensor2):
    return fnorm(tensor1-tensor2, fft=False)