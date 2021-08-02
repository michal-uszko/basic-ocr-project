# Basic Optical Character Recognition
## General Info
The goal of this project was to make a neural network capable of recognizing handwrittens letters and numbers from video stream.

For image classifier, the neural network uses ResNet50 architecture. For detecting handwritten signs on video stream, Canny edge detection algorithm was used. 

## Datasets
The neural network was learning on combination of few datasets:
* For numbers (0-9): MNIST dataset inclued in Keras library (```from tensorflow.keras.datasets import mnist```)
* For letters (A-Z): https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format
* For '@' sign: https://www.kaggle.com/vaibhao/handwritten-characters

## Installation
For this project the following libraries were used:
* OpenCV (installation: https://pypi.org/project/opencv-python/)
* Tensorflow 2.x (installation: https://www.tensorflow.org/install)
* Keras (included with Tensorflow)
* Pandas (installation: https://pandas.pydata.org/docs/getting_started/install.html)
* imutils (installation: ```pip install imutils```)
* splitfolders (installation: https://pypi.org/project/split-folders/)

All dependencies are included in ```requirements.txt``` file.

Instead of installing all libraries manually, you can also try to setup your environment by installing all dependencies from this text file. To do so run the command below:
```
pip install -r requirements.txt
```

## Setup
After installing all necessary packages and downloading datasets, you will need to create the dataset, on which the image classifier is going to learn. To do so,
run these two commands first (the order between these two commands is irrelevant):
```
python mnist_to_img.py
```
```
python az_to_img.py
```

Also, remember to copy and paste '@' folder with images to ```dataset/``` path.

Then, you need to split the images into training and test datasets. To do so, run the command below:
```
python data_split.py
```

Now, you can train the model:
```
python ResNet50_nn.py
```

After training, you can check the results with this command (you will need to connect a webcam to do so):
```
python main.py -m model\classifier.h5 -l model\labels.pickle
```
