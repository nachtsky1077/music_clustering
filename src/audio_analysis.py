from src.au_dataloader import MusicLoader
from src.mat_svd_analysis import plot_heatmap, spectral_analysis
from spectral_density import *
from seaborn import heatmap
import numpy as np
from sklearn.preprocessing import normalize

data_base_path = 'dataset_music/music_genres_dataset_small'

ml = MusicLoader(data_base_path=data_base_path, verbose=0)
music_ts = ml.fetch_data(genres=['classical', 'pop'],
                         n_examples=10,
                         n_frames=300)

# get correlation, check whether there's 'block' phenomenon
corr = np.corrcoef(music_ts)
corr_fig = plot_heatmap(corr)

# spectral analysis
kwargs = {'selected_freq_index': range(0, 151)}
au_spec_ori = spectral_analysis(music_ts.T, **kwargs)
spec_fig_freq_zero_ori = plot_heatmap(abs(au_spec_ori[0]))
spec_fig_freq_half_ori = plot_heatmap(abs(au_spec_ori[50]))

music_ts_normalize = normalize(music_ts, norm='l1')
au_spec_ori_normalized = spectral_analysis(music_ts_normalize.T, **kwargs)
spec_fig_ori_normalized = plot_heatmap(abs(au_spec_ori_normalized[75]))
spec_fig_ori_normalized.show()