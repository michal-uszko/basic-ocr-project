import splitfolders
import shutil

# Defining an input path, which is also an output path
input_path = "dataset"

# Splitting dataset into training set (80%) and test set (20%)
print("[INFO] Splitting into training and test set...")
splitfolders.ratio(input=input_path, output=input_path, seed=42, ratio=(0.8, 0, 0.2))

# Removing undivided data
print("[INFO] Deleting unnecessary folders...")
folder_names = "0123456789@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
folders = [f for f in folder_names]

for folder in folders:
    shutil.rmtree(f"dataset/{folder}", ignore_errors=True)

# Removing empty 'val' folder
shutil.rmtree("dataset/val")
