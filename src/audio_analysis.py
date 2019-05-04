from au_dataloader import MusicLoader
from mat_svd_analysis import SVDAnalyzer, plot_singular_values, plot_heatmap, spectral_analysis, fnorm
from spectral_density import *
from seaborn import heatmap
import numpy as np
from sklearn.preprocessing import normalize
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

'''
# auto covariance
auto_cov_mat = dict()
figs = []
for k in range(10, 11):
    heatmap = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            x1 = music_ts[i]
            x2 = music_ts[j]
            auto_cov_mat[(i, j, k)] = autocovariance(x1, x2, k, 8000)
            heatmap[i][j] = auto_cov_mat[(i, j, k)]
    figs.append(plot_heatmap(heatmap))
for fig in figs:
    fig.show()
'''

def music_spectral_analysis(music_ts, freq_idx_list, k, **kwargs):
    '''
    :music_ts: the time series data, pxN (p: number of music clips, N: number of observations)
    :freq_idx_list: the frequency index to perform spectral density
    :k: the number of top energy frequency picked to represent the results, when n_top=None, it should return
            all the frequency results
    :kwargs: the arguments passed down to spectral analysis function
    '''
    kwargs['selected_freq_index'] = freq_idx_list
    au_spec = spectral_analysis(music_ts.T, **kwargs)

    if k is None:
        #TODO
        return au_spec
    else:
        fnorms = []
        au_spec_modulus = {}
        # calculate fnorm of the spectral density estimation
        for freq_idx in freq_idx_list:
            au_spec_modulus[freq_idx] = abs(coherence(au_spec[freq_idx]))
            fnorms.append((freq_idx, fnorm(au_spec_modulus[freq_idx], fft=False)))
        fnorms.sort(key=lambda item: item[1], reverse=True)
        print(fnorms)
        # averaging top k frequencies
        ave = np.zeros(au_spec[0].shape)
        for i in range(k):
            ave += au_spec_modulus[fnorms[i][0]]
        ave /= k
        return ave

'''
for freq_idx in range(0, 500, 5):
    fig = plot_heatmap(abs(coherence(au_spec_ori[freq_idx])))
    fig.savefig('results/music_classical_pop/spectral_density_20x20_freq_idx_{}.png'.format(freq_idx), dpi=280)
    plt.close()
'''

'''
# svd analysis
sa = SVDAnalyzer(music_ts)
sa.analyze()
fig = plot_singular_values([sa.singular_values()], ['1'])
fig.show()
'''

if __name__ == '__main__':
    data_base_path = 'dataset_music/music_genres_dataset_tiny'

    ml = MusicLoader(data_base_path=data_base_path, verbose=0)
    music_ts = ml.fetch_data(genres=['classical', 'pop'],
                             n_examples=20,
                             n_frames=1000,
                             downsample_ratio=10)
    spec_density = music_spectral_analysis(music_ts, range(50), 10)
    fig = plot_heatmap(spec_density)
    #fig.savefig('results/heatmaps/spectral_density_average_top_{}.png'.format(10), dpi=280)
    # get correlation, check whether there's 'block' phenomenon
    #corr = np.corrcoef(music_ts)
    #corr_fig = plot_heatmap(corr)