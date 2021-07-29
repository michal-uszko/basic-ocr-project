import os
import pandas as pd
import numpy as np
from helpers import load_mnist_dataset, rename_files
from keras_preprocessing import image

if __name__ == '__main__':
    # Making directory for MNIST dataset
    mnist_dir = "dataset/"

    if not os.path.exists(mnist_dir):
        os.makedirs(mnist_dir)

    # Loading MNIST dataset with labels
    nums = load_mnist_dataset()[0]
    labels = load_mnist_dataset()[1]
    labels_unique = pd.array(labels).unique()

    # Making sub-folders for each label (from 0 to 9)
    for label in labels_unique:
        new_path = mnist_dir + f"{label}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    # Save MNIST data as images to corresponding columns
    count = 1
    for num, lb in zip(nums, labels):
        one_num = num.reshape(28, 28, 1)
        one_num = np.expand_dims(one_num, axis=0)
        save_path = mnist_dir + f"{lb}/{count}.png"
        image.array_to_img(one_num[0]).save(save_path)
        count += 1

    # Renaming files
    for i in labels_unique:
        rename_files(i, mnist_dir + f"{i}/")
