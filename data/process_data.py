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


def plot_spectrogram():
    plt.figure(0, figsize=(12, 12))
    for i in range(1, 4):
        plt.subplot(1, 3, i)
        img = np.squeeze(x_train[np.random.randint(0, x_train.shape[0])])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

    plt.imshow(np.squeeze(x_train[3]), cmap="gray");