import gzip
import idx2numpy
import numpy as np
from sklearn.model_selection import train_test_split


def load_fashion_mnist():
    with gzip.open("data/train-images-idx3-ubyte.gz", "rb") as f:
        X_train = idx2numpy.convert_from_file(f)

    with gzip.open("data/train-labels-idx1-ubyte.gz", "rb") as f:
        y_train = idx2numpy.convert_from_file(f)

    with gzip.open("data/t10k-images-idx3-ubyte.gz", "rb") as f:
        X_test = idx2numpy.convert_from_file(f)

    with gzip.open("data/t10k-labels-idx1-ubyte.gz", "rb") as f:
        y_test = idx2numpy.convert_from_file(f)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test


def get_flattened_data():
    X_train, y_train, X_test, y_test = load_fashion_mnist()

    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]