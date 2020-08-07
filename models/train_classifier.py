import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from livelossplot.tf_keras import PlotLossesCallback

sys.path.append('../')
from data import *


def sequential_model():
    # Initialising the CNN
    model = Sequential()

    # 1st Convolution
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(64, 128, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4, activation='softmax'))

    return model


def model_compile(model):
    initial_learning_rate = 0.005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=5,
        decay_rate=0.96,
        staircase=True)

    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def model_train(model, datagen_train, datagen_val, x_train, y_train, x_val, y_val):
    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_loss',
                                 save_weights_only=True, mode='min', verbose=0)
    callbacks = [PlotLossesCallback(), checkpoint]  # , reduce_lr]
    batch_size = 32
    history = model.fit(
        datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
        steps_per_epoch=len(x_train) // batch_size,
        validation_data=datagen_val.flow(x_val, y_val, batch_size=batch_size, shuffle=True),
        validation_steps=len(x_val) // batch_size,
        epochs=12,
        callbacks=callbacks
    )


def main():
    print("Loading pickle files...")

    x_train, x_val, y_train, y_val = load_images()

    print("Creating data generators...")

    datagen_train, datagen_val = create_data_generators(x_train, x_val)

    model = sequential_model()

    print("Compiling model...")

    model = model_compile(model)

    print("Training model...")

    model_train(model, datagen_train, datagen_val, x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    main()
