"""
model.py: Get Model
"""
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


def conv_block(num_filter, dropout=0.5):
    """
    Convolutional Block
    Parameters:
        num_filter: num of filter
        dropout: probability of dropout
    return:
        conv_layer
    """
    conv_layer = keras.Sequential([
        Conv2D(num_filter, (3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(),
        Dropout(dropout)
    ])
    return conv_layer


def get_model(input_shape, class_num, dropout=0.5):
    """
    Get Model
    Parameters:
        input_shape: shape of input 
        class_num: num of classes
        dropout: probability of dropout
    return:
        model
    """
    model = keras.Sequential([
        # Input Layer
        Input(shape=input_shape),
        # Convenlutional Layer
        conv_block(16),
        conv_block(32),
        conv_block(64),
        conv_block(128),
        conv_block(256),
        Flatten(),
        Dropout(dropout),
        Dense(512, activation='relu'),
        Dropout(dropout),
        Dense(class_num, activation='softmax')
    ])
    return model
