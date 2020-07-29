import pandas as pd


def load_images():
    train_images = pd.read_csv('train/images.csv', header=None)
    train_labels = pd.read_csv('train/labels.csv', header=None)

    val_images = pd.read_csv('validation/images.csv', header=None)
    val_labels = pd.read_csv('validation/labels.csv', header=None)

    x_train = train_images.values.reshape(3200, 64, 128, 1)
    x_val = val_images.values.reshape(800, 64, 128, 1)

    y_train = train_labels.values
    y_val = val_labels.values
