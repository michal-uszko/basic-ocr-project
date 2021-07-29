import os
import numpy as np
import pandas as pd
from helpers import load_az_dataset, rename_files
from keras_preprocessing import image


if __name__ == '__main__':
    # A-Z CSV dataset path file
    az_path = "A_Z Handwritten Data.csv"

    # Making directory for A-Z dataset
    az_dir = "dataset/"

    if not os.path.exists(az_dir):
        os.makedirs(az_dir)

    # Loading A-Z dataset with labels
    letters = load_az_dataset(az_path)[0]
    labels = load_az_dataset(az_path)[1]

    # Mapping labels to corresponding letters
    labels = pd.Series(labels)

    map_labl_dict = {0: "A", 1: "B", 2: "C", 3: "D",
                     4: "E", 5: "F", 6: "G", 7: "H",
                     8: "I", 9: "J", 10: "K", 11: "L",
                     12: "M", 13: "N", 14: "O", 15: "P",
                     16: "Q", 17: "R", 18: "S", 19: "T",
                     20: "U", 21: "V", 22: "W", 23: "X",
                     24: "Y", 25: "Z"}

    labels = labels.map(map_labl_dict)
    labels_unique = pd.array(labels).unique()

    # Making sub-folders for each label (from A to Z)
    for label in labels_unique:
        new_path = az_dir + f"{label}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    # Saving A-Z data as images to corresponding sub_folders
    count = 1
    for letter, lb in zip(letters, labels):
        one_letter = letter.reshape(28, 28, 1)
        one_letter = np.expand_dims(one_letter, axis=0)
        save_path = az_dir + f"{lb}/{count}.png"
        image.array_to_img(one_letter[0]).save(save_path)
        count += 1

    # Renaming files
    for i in labels_unique:
        rename_files(i, az_dir + f"{i}/")
