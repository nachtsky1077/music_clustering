from ucr_dataset_load import get_data
import matplotlib.pyplot as plt
import numpy as np
from math_utilities.mat_svd_analysis import SVDAnalyzer, plot_singular_values, plot_heatmap, spectral_analysis
from spectral_density import *
from math_utilities.util import coherence, fnorm

train_data, train_label, test_data, test_label = get_data('ChlorineConcentration')

train_data = np.asarray(train_data)
train_label = np.asarray(train_label)

def ts_spectral_analysis(ts, freq_idx_list, k, **kwargs):
    '''
    :ts: the time series data, pxN (p: number of ts clips, N: number of observations)
    :freq_idx_list: the frequency index to perform spectral density
    :k: the number of top energy frequency picked to represent the results, when n_top=None, it should return
            all the frequency results
    :kwargs: the arguments passed down to spectral analysis function
    '''
    kwargs['selected_freq_index'] = freq_idx_list
    kwargs['mode'] = 'al'
    au_spec = spectral_analysis(ts.T, **kwargs)

    au_spec_modulus = {}
    # calculate fnorm of the spectral density estimation
    for freq_idx in freq_idx_list:
        au_spec_modulus[freq_idx] = abs(au_spec[freq_idx])
    if k is None:
        return au_spec_modulus, None
    else:
        fnorms = []
        for freq_idx in freq_idx_list:
            fnorms.append((freq_idx, fnorm(au_spec_modulus[freq_idx], fft=False)))
            #fnorms.append((freq_idx, au_spec_modulus[freq_idx].ravel().std()))
        fnorms.sort(key=lambda item: item[1], reverse=True)
        #print(fnorms)
        # averaging top k frequencies
        ave = np.zeros(au_spec[0].shape)
        for i in range(k):
            ave += au_spec_modulus[fnorms[i][0]]
        ave /= k
        return au_spec_modulus, ave
    
top_k = 20
spec_density_all, spec_density = ts_spectral_analysis(train_data, freq_idx_list=[0], k=top_k)
for freq_idx in range(0, 1, 20):
    fig = plot_heatmap(spec_density_all[freq_idx])
    fig.savefig('results/heatmaps/ChlorineConcentration/adaptivelasso_spectral_density_freq_idx_{}.png'.format(freq_idx), dpi=150)
    plt.close()