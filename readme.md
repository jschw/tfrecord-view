# tfrecord-view for TF 2.x with pyQt GUI
This is a fork of the project https://github.com/EricThomson/tfrecord-view featuring a minimalistic GUI which allows browsing through the images in the dataset.
It is also updated to work with Tensorflow >= 2.0 .

![Browse example dataset](/example_dataset/TFR_View_Example.gif)

Runs fine on macOS and Windows and probably Linux.

## What are TFRecord files?
A TFRecord file is a binary format to store Tensorflow datasets in an optimized way to achieve higher performance for read operations during training (https://www.tensorflow.org/guide/data_performance). It is widely used for object detection datasets but in general, lots of other datatypes can be saved this way.

## Browsing a TFRecord file
The fileformat is binary, so you can't easily open or unzip it. But sometimes, you may want to take look inside and check, if preprocessing steps were successful. Or just determine what the file contains if you have chosen a meaningless filename... ;)

## Requirements
- Tensorflow >= 2.0
- numpy
- opencv-python
- pyqt5
- Enough RAM!

The last point is important because this script loads all images at once into the memory. Images are represented by numpy tensors of float32 type. So ~70 MB of compressed image data can fill up over 1 GB of RAM! You can significantly reduce memory consumption by setting a scale factor.

## Usage
Simply execute the script:

`python3 tfrecord_view_gui.py`

Just point to the .record file with the file choose dialog after startup. You also can point to a .pbtxt labelmap which contains all the label names used in the dataset. But this is optional. If no .pbtxt file was opened, it displays only the bounding boxes without class labels.

As mentioned above, you can set a scale factor:

`python3 tfrecord_view_gui.py -s 0.75`

This scales down all images to 75% of their original size.


