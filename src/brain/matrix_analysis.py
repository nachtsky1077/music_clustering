import os
import sys
sys.path.append(os.getcwd())
from src.math_utilities.mat_svd_analysis import SVDAnalyzer, plot_singular_values
from src.math_utilities.util import fnorm, matrix_estimation_error
from neural_dataloader import MatLoader

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

if __name__ == '__main__':
    mat_base_path = './dataset_mental/FuncTimeSeries 1/FuncTimeSeries/'
    ori_vol, est_vol, fig = svd_estimation_neural(mat_base_path, 'NIH-101_20100329_ts.mat', rank=40)
    print(matrix_estimation_error(ori_vol, est_vol, fft=False))