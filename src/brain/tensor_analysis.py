from neural_dataloader import  MatLoader
from config import config
data_loader = MatLoader(config['DATA_DIR'])

data_loader.load_mat_data('NIH-101_20100329_ts.mat')
data = data_loader.get_mat_data('vol')
print(data)

