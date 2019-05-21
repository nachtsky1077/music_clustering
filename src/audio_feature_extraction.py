import os
import numpy as np
import librosa.display
import librosa.core
import matplotlib.pyplot as plt

genres = ['classical', 'jazz', 'pop']
index = 0

for genre in genres:
    for index in range(10):
        path = 'dataset_music/music_genres_dataset/{}'.format(genre)
        filename = '{}.{}.au'.format(genre, str(index).zfill(5))

        y, sr = librosa.core.load(os.path.join(path, filename), sr=22050)
        D = np.abs(librosa.core.stft(y))
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
        plt.title('Power spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig('results/power_spectrogram_{}.png'.format(filename), dpi=280)
        plt.close()
