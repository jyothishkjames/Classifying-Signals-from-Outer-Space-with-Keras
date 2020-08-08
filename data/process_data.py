import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images():
    train_images = pd.read_csv('dataset/train/images.csv', header=None)
    train_labels = pd.read_csv('dataset/train/labels.csv', header=None)

    val_images = pd.read_csv('dataset/validation/images.csv', header=None)
    val_labels = pd.read_csv('dataset/validation/labels.csv', header=None)

    x_train = train_images.values.reshape(3200, 64, 128, 1)
    x_val = val_images.values.reshape(800, 64, 128, 1)

    y_train = train_labels.values
    y_val = val_labels.values

    return x_train, x_val, y_train, y_val


def save_pkl_files(x_train, x_val, y_train, y_val):
    with open('data.pkl', 'w') as f:
        pickle.dump([x_train, x_val, y_train, y_val], f)


def plot_spectrogram(x_train, ):
    plt.figure(0, figsize=(12, 12))
    for i in range(1, 4):
        plt.subplot(1, 3, i)
        img = np.squeeze(x_train[np.random.randint(0, x_train.shape[0])])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

    plt.imshow(np.squeeze(x_train[3]), cmap="gray")


def create_data_generators(x_train, x_val):
    datagen_train = ImageDataGenerator(horizontal_flip=True)
    datagen_train.fit(x_train)

    datagen_val = ImageDataGenerator(horizontal_flip=True)
    datagen_val.fit(x_val)

    return datagen_train, datagen_val


def main():

    print("Loading data...")

    x_train, x_val, y_train, y_val = load_images()

    print("Saving data as pickle file...")

    save_pkl_files(x_train, x_val, y_train, y_val)


if __name__ == '__main__':
    main()
