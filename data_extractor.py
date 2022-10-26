"""
Data Extractor: Extracte Feature from Raw Data
"""
import tensorflow as tf
import os
import numpy as np
import pdb
import librosa
import pickle

from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical


N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
WINDOW = 0.05
OVERLAP = 0.5
AUDIO_RANGE = 660000
SPLIT_RATIO = 0.8
SEED = 20


def conver2spectrum(sub_audios, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Convert Sub_audios to Spectrums
    Parameters:
        sub_audios: array of sub_audio
        n_fft: length of the FFT window
        hop_length: number of samples between successive frames
        n_mels: number of Mel bands to generate
    return:
        spectrums: array of spectrum
    """
    spectrums = []
    for sub_audio in sub_audios:
        spectrum = librosa.feature.melspectrogram(sub_audio, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        spectrum = np.expand_dims(spectrum, -1)
        spectrums.append(spectrum)
    spectrums = np.array(spectrums)
    return spectrums


def split_audio(y, label, window=WINDOW, overlap=OVERLAP):
    """
    Split Audio via Overlapping Windows
    Parameters:
        y: the first output of librosa.load
        label: label
        window: window ratio
        overlap: overlap ratio
    return:
        sub_audios: array of sub_audio
        labels: array of label
    """
    sub_audios = []
    labels = []

    audio_length = y.shape[0]
    sub_audios_length = int(audio_length * window)
    overlap_length = int(sub_audios_length * overlap)
    for i in range(0, audio_length - sub_audios_length + overlap_length, overlap_length):
        sub_audio = y[i:i + sub_audios_length]
        if sub_audio.shape[0] != sub_audios_length:
            continue
        sub_audios.append(sub_audio)
        labels.append(label)

    sub_audios = np.array(sub_audios)
    labels = np.array(labels)
    return sub_audios, labels


def convert2xy(file_list, label_list, audio_range=AUDIO_RANGE):
    """
    Convert file_list and label_list to x and y
    Parameters:
        file_list: file list
        label_list: label list
    return:
        x: input x
        y: target y
    """
    x_list = []
    y_list = []
    assert len(file_list) == len(label_list)
    for file, label in zip(file_list, label_list):
        y, _ = librosa.load(file)
        y = y[:audio_range]
        sub_audios, labels = split_audio(y, label)
        spectrums = conver2spectrum(sub_audios)
        x_list.extend(spectrums)
        y_list.extend(labels)
    assert len(x_list) == len(y_list)
    res_x = np.array(x_list)
    res_y = to_categorical(y_list)
    return res_x, res_y


def get_data(data_path, genres_dict, split_ratio=SPLIT_RATIO, seed=SEED):
    """
    Get Training Data & Testing Data
    Parameters:
        data_path: data path
        genres_dict: genres map dict
        split_ratio: split for training and testing (default: 0.8)
        seed: random seed number (defaul: 20)
    return:
        train_x: x for training
        train_y: y for training
        test_x: x for testing
        test_y: y for testing
    """
    # list for train_file, train_label, test_file, test_label
    train_file_list = []
    train_label_list = []
    test_file_list = []
    test_label_list = []
    # get file name list from data_path
    for genre, _ in genres_dict.items():
        genre_path = f"{data_path}/{genre}"
        for _, _, file_names in os.walk(genre_path):
            length = len(file_names)
            genre_label = genres_dict[genre]
            genre_labels = []
            split_index = round(length * split_ratio)
            # get full path
            file_list = []
            for file_name in file_names:
                res = f"{genre_path}/{file_name}"
                # filter invalid data
                if file_name == "jazz.00054.wav":
                    continue
                file_list.append(res)
                genre_labels.append(genre_label)
            # split for train & test 
            train_file_list += file_list[:split_index]
            train_label_list += genre_labels[:split_index]
            test_file_list += file_list[split_index:]
            test_label_list += genre_labels[split_index:]

    # shufflue
    train_file_list, train_label_list = shuffle(train_file_list, train_label_list, random_state=seed)
    test_file_list, test_label_list = shuffle(test_file_list, test_label_list, random_state=seed)

    # convert to input
    train_x, train_y = convert2xy(train_file_list, train_label_list)
    test_x, test_y = convert2xy(test_file_list, test_label_list)
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    # Extracte Data
    GENRES_MAP = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
    }
    SEED = 20
    SPLIT_RATIO = 0.8
    DATA_PATH = "./Data/genres_original"
    train_x, train_y, test_x, test_y = get_data(DATA_PATH, GENRES_MAP, SPLIT_RATIO, SEED)
    
    # Save Data
    data_dict = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y
    }
    OUT_PATH = "data_dict.pkl"
    with open(OUT_PATH, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"Finish Data Extraction and Extracted Data is {OUT_PATH}")
