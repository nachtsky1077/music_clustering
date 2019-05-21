import numpy as np
from librosa import cqt
import librosa.core
import pywt

'''
Surface features:
 1. mean-Centroid
 2. mean-Rolloff
 3. mean-Flux
 4. mean-ZeroCrossings
 5. variance-Centroid
 6. variance-Rolloff
 7. variance-Flux
 8. variance-ZeroCrossings
 9. LowEnergy

mean and variance are calculated over a "texture" window of 1 second consisting of 40 "analysis" windows of
20 milliseconds (512 samples at 22050 sampling rate). The feature calculation is based on STFT.

'''

def centroid(spec, fft_freq):
    '''
    The Centroid is a measure of spectral brightness.
    :spec: the magnitude of the FFT, shape 1xN (where N is the number of FFT bins)
    :fft_freq: the fft frequency list, it should be in shape 1xN
    '''
    spec_mag = np.abs(spec)
    weighted_sum = (spec_mag * fft_freq).sum()
    c = weighted_sum / spec_mag.sum()
    return c

def rolloff(spec, rolloff_percent=0.85):
    '''
    The rolloff is a measure of spectral shape.
    :spec: the magnitude of the FFT, shape 1xN (where N is the number of FFT bins)
    '''
    spec_mag = np.abs(spec)
    target_sum = spec_mag.sum() * rolloff_percent
    curr_sum = 0
    for i in range(spec.size):
        if curr_sum <= target_sum and curr_sum+spec_mag[i] >= target_sum:
            r = i
            break
        curr_sum += spec_mag[i]
    return r

def flux(spec, prev_spec):
    '''
    The flux is a measure of spectral change.
    F = ||M[f] - M_p[f]||
    Use 2-norm.
    :spec: the magnitude of the FFT, shape 1xN (where N is the number of FFT bins)
    :prev_spec: the magnitude of the FFT of the previous frame, shape 1xN (where N is the number of FFT bins)
    '''
    spec_mag = np.abs(spec)
    if not prev_spec is None:    
        prev_spec_mag = np.abs(prev_spec)
        diff = spec_mag - prev_spec_mag
    else:
        diff = spec_mag
    f = np.linalg.norm(diff, ord=2)
    return f

def zero_crossings(ts):
    '''
    The number of time domain zero crossings of the signal.
    ZeroCrossings are useful to detect the amount of noise in signal.
    :ts: the time series of one frame
    '''
    indices = np.where(np.diff(np.signbit(ts)))
    return indices[0].size

def low_energy(ts_texture):
    '''
    The percentage of "analysis" windows that have energy less than the average energy of the "analysis"
    windows over the "texture" window.
    :ts_texture: a list of ts for each "analysis" window
    '''
    rmse = np.array([np.sqrt(np.mean(np.square(ts))) for ts in ts_texture])
    ave_rmse = rmse.mean()
    le = (rmse < ave_rmse).sum()
    return le


'''
TODO:
Rhythm Features:

First decompose the signal into a number of octave frequency bands using DWT, then the time domain amplitude
envelope of each band is extracted separately. This is ahieved by applying full wave rectification, low pass
filtering and downsampling to each band. The envelopes of each band are then summed together and an autocorrelation
function is computed. The peaks of the autocorrelation function correspond to the various periodicities of the
signal's envelope.
'''

def octave_decomp(ts, sr):
    c = np.abs(cqt(ts, sr=sr))
    return c