import os
import numpy as np
import sunau
import matplotlib.pyplot as plt

class AuLoader(object):

    np_dtype = ['b', '>h']

    def __init__(self, filename):
        self.au_read = sunau.open(filename, 'r')
        # get file parameters
        self.params = self.au_read.getparams()
        
    def load(self, nframes = None):
        '''
        :nframes: the number of frames to be loaded from the audio file
        '''
        nframes = min(nframes, self.params.nframes) if nframes is not None else self.params.nframes
        
        # read nframes from the au file reader
        bytes_stream = self.au_read.readframes(nframes)
        
        # covert the bytes stream into a numpy array
        # notes: the byte stream is in big endian so we use '>h'
        self.data = np.fromstring(bytes_stream, dtype = '>h')

        return self.data
    
    def audio_params(self):
        return self.params


class MusicLoader(object):

    def __init__(self, data_base_path, verbose=0):
        '''
        :data_base_path: the base path where the dataset locates
        :verbose: the verbose level
        '''
        self.base_path = data_base_path
        self.verbose = verbose

    def print(self, level, message):
        if level <= self.verbose:
            print(message)

    def fetch_data(self, genres, n_examples=20, n_frames=100):
        '''
        :genres: a list of genres to fetch data from, genre should match a folder name
        :n_observation: the number observations to extract from the music time series
        '''
        if type(genres) != type([]):
            categories = [genres]
        ts_data = None
        for genre in genres:
            base_path = os.path.join(self.base_path, genre)
            self.print(3, 'base_path:{}'.format(base_path))
            for i in range(n_examples):
                filename = '{}.{}.au'.format(genre, str(i).zfill(5))
                file_full_path = os.path.join(base_path, filename)
                self.print(3, 'file full path:{}'.format(file_full_path))
                au_loader = AuLoader(file_full_path)
                data = au_loader.load(n_frames)
                data = data.reshape(1, data.shape[0])
                self.print(3, 'data shape:{}'.format(data.shape))
                self.print(3, 'data:{}'.format(data))
                if ts_data is None:
                    ts_data = np.array(data)
                else:
                    ts_data = np.concatenate([ts_data, data], axis=0)
        self.print(2, 'ts data matrix:{}'.format(ts_data))
        return ts_data
        
            


'''
music_sample_file = 'data/genres/country/country.00000.au'
test_sample_file = 'data/1khz_test/1kHz_44100Hz_16bit_05sec.au'

au_loader = AuLoader(test_sample_file)
nFrames = 500
sampRate = au_loader.audio_params().framerate
T =  1.0 / sampRate
data = au_loader.load(nFrames)

data_fft = np.fft.fft(data)
x_fft = np.linspace(0.0, 1.0 / (2.0 * T), nFrames / 2)
plt.plot(x_fft, 2.0 / nFrames * np.abs(data_fft[0 : int(nFrames / 2)]))
plt.grid()
plt.show()
'''

