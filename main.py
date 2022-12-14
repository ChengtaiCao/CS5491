"""
Main Function
"""
import pickle
import argparse
import numpy as np
import pdb

from model import *
from mixup import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=int, required=True, choices=[0, 1])
    args = parser.parse_args()
    if args.aug == 1:
        aug_flag = True
        str_text = "with aug"
    else:
        aug_flag = False
        str_text = "without aug"
    NUM_CLASS = 10

    # Get data
    PATH = "data_dict.pkl"
    with open(PATH, "rb") as f:
        data_dict = pickle.load(f)
    train_x = data_dict["train_x"]
    train_y = data_dict["train_y"]
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]

    # Split validation
    num_train = train_x.shape[0]
    split_index = round(num_train * 0.8)
    splited_train_x = train_x[:split_index]
    splited_train_y = train_y[:split_index]
    splited_validation_x = train_x[split_index:]
    splited_validation_y = train_y[split_index:]

    # Model
    input_shape = train_x[0].shape
    model = get_model(input_shape, NUM_CLASS)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])

    # Reduce learning rate when a metric has stopped improving.
    reduceLROnPlat = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.9,
        patience=3,
        verbose=1,
        mode='min',
        min_delta=0.0001,
        cooldown=2,
        min_lr=0.00001
    )

    BATCH_SIZE = 128
    # BatchDataset 
    
    if not aug_flag:
        # No augmentation
        train_ds = (
            tf.data.Dataset.from_tensor_slices((splited_train_x, splited_train_y))
            .batch(BATCH_SIZE)
        )
    else:
        # Mixup augmentation
        train_ds_one = (
            tf.data.Dataset.from_tensor_slices((splited_train_x, splited_train_y))
            .shuffle(BATCH_SIZE * 100)
            .batch(BATCH_SIZE)
        )
        train_ds_two = (
            tf.data.Dataset.from_tensor_slices((splited_train_x, splited_train_y))
            .shuffle(BATCH_SIZE * 100)
            .batch(BATCH_SIZE)
        )
        train_ds_origin = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds = train_ds_origin.map(
            lambda ds_one, ds_two: mixup(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
        )
        
    validation_ds = (
        tf.data.Dataset.from_tensor_slices((splited_validation_x, splited_validation_y))
        .batch(BATCH_SIZE)
    )
    # Train
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=200,
        verbose=1,
        callbacks=[reduceLROnPlat])
    # Save Model
    SAVE_PATH = "model.h5"
    model.save(SAVE_PATH)
    # Test
    score = model.evaluate(test_x, test_y, verbose=0)
    print(f"CNN mean accuracy {str_text}: {score[1]:.4f}")

    plt.figure(figsize=(15,7))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
