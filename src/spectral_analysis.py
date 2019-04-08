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
        self.data = np.fromstring(bytes_stream, dtype = '>h')

        return self.data
    
    def audio_params(self):
        return self.params


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


