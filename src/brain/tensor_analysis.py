from neural_dataloader import  MatLoader
from config import config
import numpy as np
data_loader = MatLoader(config['DATA_DIR'])

data_loader.load_mat_data('NIH-101_20100329_ts.mat')
data = data_loader.get_mat_data('vol')
print(data.shape)



def stack(nums):
    '''
    :param nums: number of the mri data to stack
    :return: a tensor
    86*200*num
    '''
    tensor = np.zeros((1,1,nums))   #
    for i in range():





def tucker_decomp(tensor, rank, sketch=False):
    '''
    :return: tucker decomposition, and approx error
    '''

    #return .... , error



