from src.au_dataloader import MusicLoader
from src.mat_svd_analysis import SVDAnalyzer, plot_singular_values, plot_heatmap, spectral_analysis
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

data_base_path = 'dataset_music/music_genres_dataset'

ml = MusicLoader(data_base_path=data_base_path, verbose=0)
music_ts = ml.fetch_data(genres=['classical', 'pop'],
                         n_examples=100,
                         n_frames=20000,
                         downsample_ratio=10)

# get correlation, check whether there's 'block' phenomenon
corr = np.corrcoef(music_ts)
corr_fig = plot_heatmap(corr)
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

# spectral analysis
kwargs = {'selected_freq_index': range(0, 500)}
au_spec_ori = spectral_analysis(music_ts.T, **kwargs)
fig = plot_heatmap(abs(coherence(au_spec_ori[50])))


# svd analysis
sa = SVDAnalyzer(music_ts)
sa.analyze()
fig = plot_singular_values([sa.singular_values()], ['1'])
fig.show()