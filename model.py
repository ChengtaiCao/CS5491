"""
model.py: Get Model
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model



def conv_block(x, num_filter, dropout=0.5):
    """
    Convolutional Block
    Parameters:
        x: input
        num_filter: num of filter
        dropout: probability of dropout
    return:
        output of convolutional block
    """
    x = Conv2D(num_filter, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(dropout)(x)
    return x


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
    inpt = Input(shape=input_shape)
    # Convenlutional Layer
    x = conv_block(inpt, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    # Flatten
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout)(x)
    # Output
    predictions = Dense(class_num, activation='softmax')(x)
    
    model = Model(inputs=inpt, outputs=predictions)
    return model
