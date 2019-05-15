# ASL Recognition from the Webcam

This repository is an implementation of using CNN network to identify American Sign Language gestures from the webcam and show the output on the screen.

**Demo:**

[![Demo Video](https://github.com/ruslan-kl/asl_recognition/blob/master/misc/demo_video.jpg?raw=true)](https://youtu.be/cbf1KNF3RJ0)

## Requirements

* `opencv-python`
* `numpy`
* `tensorflow`
* `spellchecker`


## Data Set

Data set was created by me (approximately 1200 images for 1 gesture). 

**Data set structure:**
```
|-- img_dataset
|   |-- train # train images ~80%
|       |-- A
|           |-- 0.jpg
|           |-- 1.jpg
|           |-- ...
|       |-- ...
|   |-- test # test images ~20%
|       |-- A
|           |-- 0.jpg
|           |-- 1.jpg
|           |-- ...
|       |-- ...
|   |-- orig_sample # RGB images of ASL gesture (one image for a gesture)
|       |-- A.jpg
|       |-- B.jpg
|       |-- ...
```

**American Sign Language gestures:**

![](https://github.com/ruslan-kl/asl_recognition/blob/master/misc/orig_imgs.jpg?raw=true)

## Model Training
For classification model Canny Edge Detector was applied on images.

**Images that are used for model:**

![](https://github.com/ruslan-kl/asl_recognition/blob/master/misc/model_imgs.jpg?raw=true)

**CNN model performance:**

![](https://github.com/ruslan-kl/asl_recognition/blob/master/misc/model_performance.jpg?raw=true)

[Google Colaboratory Notebook](https://colab.research.google.com/drive/1i9nmSJRXNlG8RtfRCeCXVUCJ1ovUekcF) with training process.

## Run the Script

*You might need to install the [Git Large File Storage](https://github.com/git-lfs/git-lfs/wiki/Installation) to be able to clone the repository.*

```
$ git lfs install
$ git clone https://github.com/ruslan-kl/asl_recognition.git
$ cd asl_recognition
$ pip install -r requirements.txt
$ python asl_recognizer.py
```

**Some instructions:**

* Once started adjust the threshold values for edge detection so you can see just the edges of your palm and fingers.
![](https://github.com/ruslan-kl/asl_recognition/blob/master/misc/threshold_values.gif?raw=true)
* Press `S` to start/pause the output generation.
* Press `D` to erase the output section.
* Press `Q` to quit the script.
* `del`, `space` and `nothing` do what they suppose to do.
* Input double `space` to apply `spellchecker` on the last word.
