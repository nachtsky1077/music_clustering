import os
import sys
import librosa.core
import numpy as np
sys.path.append(os.getcwd())
from sklearn.naive_bayes import GaussianNB
from src.au_dataloader import MusicLoader
from feature_utilities import *

num_texture_window = 30
num_ana_window = 100
frame_size = 512
hop_size = 64
n_examples = 3
genres = ['classical', 'pop']#, 'jazz']
num_frames = num_ana_window * num_texture_window
num_features = 9
num_feature_cat = 4
sr = 22050

# generate feature matrix for one sample
def extract_feature_matrix(sample_idx, music_data, sr=sr, frame_size=frame_size):
    feature_mat = np.zeros((num_features, num_texture_window))
    for texture_window_idx in range(num_texture_window):
        texture_window_mat = np.zeros((num_feature_cat, num_ana_window))
        prev_spec = None
        ts_texture = []
        for analysis_window_idx in range(num_ana_window):
            frame_num = texture_window_idx * num_ana_window + analysis_window_idx
            ts = music_data[frame_num][sample_idx, :]
            ts_texture.append(ts)
            # do fft first
            fft_freq = librosa.core.fft_frequencies(sr=sr, n_fft=frame_size)
            spec = np.fft.fft(ts)[:fft_freq.size]
            # centroid
            c = centroid(spec, fft_freq)
            # rolloff
            r = rolloff(spec)
            # flux
            f = flux(spec, prev_spec)
            # zero crossings
            z = zero_crossings(ts)

            texture_window_mat[0, analysis_window_idx] = c
            texture_window_mat[1, analysis_window_idx] = r
            texture_window_mat[2, analysis_window_idx] = f
            texture_window_mat[3, analysis_window_idx] = z

            prev_spec = spec

        # mean-Centroid
        feature_mat[0, texture_window_idx] = texture_window_mat[0].mean()
        # mean-Rolloff
        feature_mat[1, texture_window_idx] = texture_window_mat[1].mean()
        # mean-Flux
        feature_mat[2, texture_window_idx] = texture_window_mat[2].mean()
        # mean-ZeroCrossings
        feature_mat[3, texture_window_idx] = texture_window_mat[3].mean()
        # variance-Centroid
        feature_mat[4, texture_window_idx] = texture_window_mat[0].var()
        # variance-Rolloff
        feature_mat[5, texture_window_idx] = texture_window_mat[1].var()
        # variance-Flux
        feature_mat[6, texture_window_idx] = texture_window_mat[2].var()
        # variance-ZeroCrossings
        feature_mat[7, texture_window_idx] = texture_window_mat[3].var()
        # low energy
        feature_mat[8, texture_window_idx] = low_energy(ts_texture)
    return feature_mat

if __name__ == '__main__':
    # load music as training set
    n_examples = 30
    music_loader = MusicLoader('dataset_music/music_genres_dataset')
    music_data = music_loader.librosa_fetch(genres, 
                                            sr=sr, 
                                            n_examples=n_examples, 
                                            frame_size=frame_size, 
                                            hop_size=hop_size, 
                                            num_frames=num_frames)
    feature_matrices = dict()
    X = []
    Y = []
    train_num = 20
    for i, genre in enumerate(genres):
        for j in range(train_num):
            idx = i * n_examples + j
            feature_matrices[(genre, j)] = extract_feature_matrix(idx, music_data, sr, frame_size)
            X.append(feature_matrices[(genre, j)][0])
            Y.append(i)
    clf = GaussianNB()
    clf.fit(X, Y)

    for i, genre in enumerate(genres):
        for j in range(train_num, n_examples):
            idx = i * n_examples + j
            test_X = extract_feature_matrix(idx, music_data, sr, frame_size)
            prediction = clf.predict([test_X[0]])
            print('real class:{}, predicted class:{}'.format(i, prediction))

        
            

