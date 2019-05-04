from scipy.linalg import svd
from seaborn import heatmap
from spectral_density import *

from src.au_dataloader import *
from src.brain.neural_dataloader import *


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
            self.U_est = self.U[:,:rank]
            self.s_est = self.s[:rank]
            self.VT_est = self.VT[:rank, :]
            estimation = np.matmul(self.U_est, np.diag(self.s_est))
            estimation = np.matmul(estimation, self.VT_est)

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

#############################################################################################
# svd api
#############################################################################################
def svd_estimation_neural(mat_base_path, filename, rank=20, plot_sigma_decay=False):
    '''
    :mat_base_path: the base path for the .mat files
    :size: number of random .mat files selected to analyze
    :rank: the rank used to do svd estimation \hat{X} = svd_estimation(X, rank)
    :plot_sigma_decay: plot the singular values for each .mat matrix
    '''
    matloader = MatLoader(mat_base_path)
    sa = SVDAnalyzer()
    sigma = []
    matloader.load_mat_data(filename)
    vol = matloader.get_mat_data('vol')
    sa.mat_is(vol)
    sa.analyze()
    sigma.append(sa.singular_values())
    est_vol = sa.svd_estimation(rank=rank)

    # plot singular values decay scatter
    fig = plot_singular_values(sigma, [filename]) if plot_sigma_decay else None

    return vol, est_vol, fig

def svd_estimation_music(base_path, genres, n_files, rank, plot_sigma_decay=False):
    '''
    :base_path: music data base path
    :genres: a list contains the genres(folder) to load data from
    :n_files: specifies the number of audio files to load for each genre
    :rank:
    :plot_sigma_decay:
    '''
    # TODO
    pass

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
    selected_freq_idx = kwargs.get('selected_freq_index', None)

    spec_est = SpecEst(ts, model_info, selected_freq_index=None, individual_level=True, simu=False)

    spec_est_one_freq = {}
    for freq_idx in selected_freq_idx:
        spec_est_one_freq[freq_idx] = spec_est.query_smoothing_estimator(freq_idx)
    
    return spec_est_one_freq

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

def plot_heatmap(mat):
    fig, ax = plt.subplots()
    heatmap(mat, ax=ax)
    return fig

def estimation_relative_error(original_vol, estimated_vol, fft):
    diff = original_vol - estimated_vol
    norm_diff = fnorm(diff, fft)
    norm_ori = fnorm(original_vol, fft)
    return norm_diff, norm_diff / norm_ori

if __name__ == '__main__':
    mat_base_path = './dataset_mental/FuncTimeSeries 1/FuncTimeSeries/'
    ori_vol, est_vol, fig = svd_estimation_neural(mat_base_path, 'NIH-101_20100329_ts.mat', rank=40)
    kwargs = {'selected_freq_index': [0, 50]}

    # original ts spectral analysis
    spec_ori = spectral_analysis(ori_vol.T, **kwargs)
    fig_freq_zero_ori = plot_heatmap(abs(spec_ori[0]))
    fig_freq_half_ori = plot_heatmap(abs(spec_ori[50]))
    spec_ori_vals = spec_ori.values()
    fig_freq_ave_ori = plot_heatmap(sum([abs(mat) for mat in spec_ori_vals]) / len(spec_ori))
    # estimated ts spectral analysis
    spec_est = spectral_analysis(est_vol.T, **kwargs)
    fig_freq_zero_est = plot_heatmap(abs(spec_est[0]))
    fig_freq_half_est = plot_heatmap(abs(spec_est[50]))
    spec_est_vals = spec_est.values()
    fig_freq_ave_ori = plot_heatmap(sum([abs(mat) for mat in spec_est_vals]) / len(spec_est))

    # check relative error
    absolute_err_time_domain, relative_err_time_domain = estimation_relative_error(ori_vol, est_vol, False)
    absolute_err_freq_domain0, relative_err_freq_domain0 = estimation_relative_error(spec_ori[0], spec_est[0], False)
    absolute_err_freq_domain1, relative_err_freq_domain1 = estimation_relative_error(spec_ori[50], spec_est[50], False)





