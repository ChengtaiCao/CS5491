"""
Main Function
"""
import pickle
import numpy as np

from model import *
from data_load import *
from tensorflow.keras.callbacks import ReduceLROnPlateau


if __name__ == "__main__":
    NUM_CLASS = 10
    PATH = "data_dict.pkl"
    with open(PATH, "rb") as f:
        data_dict = pickle.load(f)

    X_train = data_dict["train_x"]
    y_train = data_dict["train_y"]
    X_test = data_dict["test_x"]
    y_test = data_dict["test_y"]

    model = get_model(X_train[0].shape, NUM_CLASS)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

    reduceLROnPlat = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.95,
        patience=3,
        verbose=1,
        mode='min',
        min_delta=0.0001,
        cooldown=2,
        min_lr=1e-5
    )

    # Generators
    batch_size = 128
    train_generator = GTZANGenerator(X_train, y_train)
    steps_per_epoch = np.ceil(len(X_train)/batch_size)

    validation_generator = GTZANGenerator(X_test, y_test)
    val_steps = np.ceil(len(X_test)/batch_size)

    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=val_steps,
        epochs=150,
        verbose=1,
        callbacks=[reduceLROnPlat])

    SAVE_PATH = "./model"
    model.save_model(SAVE_PATH)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("CNN mean accuracy:: {:.3f}".format(score[0], score[1]))
