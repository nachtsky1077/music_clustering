#!/home/yg93/anaconda3/bin/python
from au_dataloader import MusicLoader
from math_utilities.mat_svd_analysis import SVDAnalyzer, plot_singular_values, plot_heatmap, spectral_analysis
from spectral_density import *
from seaborn import heatmap
import numpy as np
from sklearn.preprocessing import normalize
from functools import reduce
from math_utilities.util import coherence, fnorm
from sklearn.cluster import SpectralClustering

def autocovariance(x1, x2, k, N=10000):
    temp = np.array([x1[k:N], x2[:N-k]])
    print(temp)
    cov = np.corrcoef(temp)
    return cov[0][1]


def coherence(spd, flag=True):
    if flag:
        D = np.real(np.diag(1.0/np.sqrt(np.diag(spd))))
        res = reduce(np.dot, [D, spd, D])
        #res[np.diag_indices(res.shape[0])] = 0
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

def spectral_clustering(affinity_matrix, n_clusters=None):
    spec_clustering = SpectralClustering(n_clusters=n_clusters,
                                         affinity='precomputed',
                                         assign_labels='discretize',
                                         random_state=0)
    spec_clustering.fit(affinity_matrix)
    return spec_clustering.labels_

def reorder_spec_density(spec_matrix, labels):
    reordered = []
    clusters = np.unique(labels)
    for cluster in clusters:
        for i in range(spec_matrix.shape[0]):
            if labels[i] == cluster:
                reordered.append(spec_matrix[i])
    return np.array(reordered)


if __name__ == '__main__':
    data_base_path = 'dataset_music/music_genres_dataset'
    top_k = 50
    ml = MusicLoader(data_base_path=data_base_path, verbose=0)
    #music_ts = ml.fetch_data(genres=['pop', 'rock'],
    #                         n_examples=10,
    #                         n_frames=20000,
    #                         downsample_ratio=10)

    num_frames = 20
    frame_size = 1024 * 8
    hop_size = 64 * 4
    music_ts_dict = ml.librosa_fetch(genres=['pop', 'rock'], sr=22050, n_examples=5, frame_size=frame_size, hop_size=hop_size, num_frames=num_frames)
    for frame_num in range(num_frames):
        music_ts = music_ts_dict[frame_num]
        spec_density_all, spec_density = music_spectral_analysis(music_ts, range(frame_size / 4), None)
    
        for freq_idx in range(0, frame_size / 4 / 2, 30):
            fig = plot_heatmap(spec_density_all[freq_idx])
            fig.savefig('results/heatmaps/spectral_density_frame_{}_freq_idx_{}.png'.format(frame_num, freq_idx), dpi=150)
            plt.close()
        #fig = plot_heatmap(spec_density)
        #fig.savefig('results/heatmaps/spectral_density_average_top_{}.png'.format(top_k), dpi=280)

    # spectral clustering
    #labels = spectral_clustering(spec_density, 2)
    #print(labels)
    #reordered_spec = reorder_spec_density(spec_density, labels)
    #reordered_fig = plot_heatmap(reordered_spec)
    #reordered_fig.savefig('results/heatmaps/spectral_density_average_top_{}_reordered.png'.format(top_k), dpi=280)
    # get correlation, check whether there's 'block' phenomenon
    #corr = np.corrcoef(music_ts)
    #corr_fig = plot_heatmap(corr)
