# Object Detection in Images
1. Object detection with HOG/SVM
1. Object detection with Yolo V3
1. Object detection with Faster R-CNN
1. Object detection with Mask R-CNN
1. Multiple object tracking with opencv-python
1. Text detection/recognition in images with EAST/Tesseract
1. Face detection with Viola-Jones/Haar-like features
* colab: https://colab.research.google.com/github/robert0714/Packt-Python-Image-Processing-Cookbook-2020/blob/main/Chapter%2008/Chapter%2008.ipynb
## Object detection with HOG/SVM
### Key words
* **Histogram of Oriented Gradients (HOG)**: HOG descriptors can be computed from an image by first computing the horizontal and vertical gradient images, then computing the gradient histograms and normalizing across blocks, and finally flattening into a feature descriptor vector. 
*  **Support Vector Machine (SVM)** :a (linear) SVM  binary classifier model is trained with several positive and negative training example images. 
#### See also
Refer to the following links to learn more about this recipe:
* https://gist.github.com/CMCDragonkai/1be3402e261d3c239a307a3346360506
* https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
* https://www.youtube.com/watch?v=kl6-NHxcn-k
* https://www.youtube.com/watch?v=qRouVgXb1G4

## Object detection with Yolo V3
* https://pjreddie.com/darknet/yolo/
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```
### How it works...
In the ``post_process()`` function, all bounding boxes (returned by the model as output) were scanned through, subsequently discarding the ones with low confidence scores. The detected object's class label was the one corresponding to the highest probability score. The non-maximum suppression algorithm was run to prune overlapping/redundant bounding boxes.

The ``opencv-python`` function, ``cv2.dnn.readNetFromDarknet()``, was used to read the pretrained Darknet model, using the paths to the provided ``.cfg`` file with a text description of the network architecture and the ``.weights`` file with a pretrained network as parameters.

The ``cv2.dnn.blobFromImage()`` function was used to create a 4D blob (the format in which the deep learning model expects its input) from the input image. The net.setInput() function was used to set the input to the network.

The ``net.forward()`` function was used to run the forward pass and obtain the outputs at the output layers.

Finally, the ``post_process()`` function was used to remove the overlapping bounding boxes and those bounding boxes with low confidence (less than the confidence threshold provided).

### There's more...
The pretrained weights were obtained by training the model with the MS-COCO image dataset (get it from here: http://cocodataset.org/) as input. Try to train the model on your own and then use the model to detect objects in some unseen images.

### See also
Refer to the following links to learn more about this recipe:

* https://pjreddie.com/darknet/yolo/
* https://arxiv.org/abs/1506.02640
* https://arxiv.org/abs/1612.08242
* https://arxiv.org/abs/1804.02767
* http://cocodataset.org/
* https://sandipanweb.wordpress.com/2018/03/11/autonomous-driving-car-detection-with-yolo-in-python/
* https://www.youtube.com/watch?v=nDPWywWRIRo
* https://www.youtube.com/watch?v=9s_FpMpdYW8
* https://www.youtube.com/watch?v=iSB_xbYA0wE
* https://www.youtube.com/watch?v=iSB_xbYA0wE
* Chapter 10 of the book, Hands-on Image Processing with Python

## Object detection with Faster R-CNN
* Reference: https://arxiv.org/pdf/1506.01497.pdf
```bash
wget https://github.com/datitran/object_detector_app/raw/master/object_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb
```
As discussed in a previous Chapter 7, *Image Segmentation*, in the *Deep instance segmentation* recipe, region-based object detection methods (for example, R-CNN and Fast R-CNN) rely on region proposal algorithms (selective search) to guess object locations. Faster R-CNN is yet another region-based object detection model that was proposed as an improvement on R-CNN (2013) and Fast R-CNN (2015), by Girshick et al. again. Fast R-CNN decreases the execution time of detection (for example, for the slower R-CNN model) by introducing ROI Pooling, but still, region proposal computation becomes a bottleneck. Faster R-CNN introduces a **Region Proposal Network (RPN)**. It achieves almost cost-free region proposals by sharing convolutional features with the detection network.

A **Region Proposal Network (RPN)** is an FCN that predicts regions that potentially contain an object with the object bounding boxes along with the objectness scores (that is, the probability that a region contains an object) at each position. End-to-end training of the RPN enables it to predict region proposals with high quality (using anchors and with recent attention mechanisms). Fast R-CNN then uses these regions for possible detection. The RPN and Fast R-CNN are concatenated to form a single network. The network is jointly trained with four losses:

* RPN provides a classification of object/not object (with an objectness score).
* RPN uses regression to compute box coordinates.
* A final classifier (from Fast R-CNN) classifies the object (with a classification score).
* Output bounding boxes corresponding to the object are computed with regression.

The RPN implements a sliding window on top of the features of the CNN. For each location of the window, it computes a score and a (per-anchor) bounding box (if k is the number of anchors, 4k bounding box coordinates are needed to be computed). Thereby, the Faster R-CNN enables object detection in a test image to be performed in real time. The https://arxiv.org/pdf/1506.01497.pdf  shows the architecture of the Faster R-CNN network.

In this recipe, you will learn how to use a pretrained Faster-RCNN model in TensorFlow to detect objects in an image.


### How it works...
For each object detected in the image, the output (out) returned (by running the forward pass with the model) contains the following:

* Bounding box rectangle coordinates for the object (``out[2][0][i]``, for the i<sup>th</sup> object)
* Class labels assigned to the object with confidence (``out[3][0][i]``, the most probable class for the i<sup>th</sup> object)
* A probability (confidence) for each class label for the object (``out[1][0][i]``, the confidence corresponding to the most probable class for the ith object)

Graph (Computational Graph) is the core concept of ``tensorflow`` to present computation. GraphDef is a serialized version of a graph and the pretrained model's ``GraphDef`` object can be parsed using the ``ParseFromString()`` function.

The ``import_graph_def()`` function can be used to import a serialized TensorFlow ``GraphDef`` protocol buffer. It extracts individual GraphDef objects as ``tf.Tensor/tf.Operation`` objects. Once extracted, these objects are placed in the current default graph.

### There's more...
Load the `tensorflow` pretrained model with `opencv-python` and run inference to detect objects in your images. Train your own Faster R-CNN model on Pascal-VOC images using a GPU (download the annotated image dataset from here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).

> To train with Keras, refer to https://github.com/kbardool/keras-frcnn. Implement object detection using the vanilla R-CNN and Fast R-CNN models, and compare the speed and precision with Faster-RCNN on a test image. 

### See also
Refer to the following links to learn more about this recipe:

* https://arxiv.org/pdf/1506.01497.pdf
* https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf
* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
* https://stackoverflow.com/questions/47059848/difference-between-tensorflows-graph-and-graphdef
* https://www.tensorflow.org/api_docs/python/tf/graph_util/import_graph_def
* http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* https://www.youtube.com/watch?v=Z-CmHOoOJJA 

## Object detection with Mask R-CNN
First, download the pretrained Mask R-CNN model from http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz (the model being trained with tensorflow on the MS-COCO dataset again with Inception v2 as the backbone network), and extract the compressed model in the appropriate path inside the models folder. 
```bash
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
wget https://github.com/spmallick/learnopencv/raw/master/Mask-RCNN/mscoco_labels.names
wget https://github.com/spmallick/learnopencv/raw/master/Mask-RCNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
wget https://github.com/spmallick/learnopencv/raw/master/Mask-RCNN/colors.txt
```
* Reference: 
  * https://github.com/spmallick/learnopencv/tree/master/Mask-RCNN
  * 

The **Mask R-CNN algorithm** (2017), by Girshick et al., includes a number of improvements compared with the Faster R-CNN algorithm for region-based object detection, with the following two primary contributions:  
* **ROI Pooling** is replaced with an **ROI Align module** (which is more accurate).
* An additional branch is inserted (which receives the output from ROI Align, subsequently feeding it into two successive convolution layers. Output from the last convolutional layer forms the object mask) at the output of the ROI Align module. 

The **RoIAlign** module provides a more precise correspondence between the regions of the feature map selected and those of the input image. Much more fine-grained alignment is needed for pixel-level segmentation, rather than just computing the bounding boxes. The https://arxiv.org/pdf/1703.06870.pdf screenshot shows the architecture of Mask R-CNN.

In this recipe, you will learn how to use a pretrained Mask R-CNN (tensorflow) model to detect objects in an image, this time using opencv-python library functions.

### How it works...
The draw_box() function does the following:

1. It draws a bounding box around a detected object.
1. It prints a label of the (most probable) class the object is assigned to.
1. It displays the label to the top-left corner of the bounding box.
1. Then, it resizes the mask, threshold, and color and applies it to the image.
1. Finally, it draws the contours on the image corresponding to the mask.

The post_proress() function extracts the bounding box along with the mask for every object detected for each image. Then, it chooses the right color (corresponding to the class label of the object) to draw the object bounding box and overlay the mask on the image.

### There's more...
Train the Mask R-CNN network on the MS-COCO dataset (download the dataset from here: http://cocodataset.org/) and save the model trained (a GPU is highly recommended); use it to predict objects in your image. 

### See also
Refer to the following links to learn more about this recipe:
* https://arxiv.org/pdf/1703.06870.pdf
* https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/
* https://github.com/facebookresearch/Detectron
* https://github.com/spmallick/learnopencv/tree/master/Mask-RCNN
* https://arxiv.org/pdf/1405.0312.pdf
* https://www.youtube.com/watch?v=g7z4mkfRjI4
* https://www.youtube.com/watch?v=FR25P1lMBY8


## Multiple object tracking with opencv-python


**Object tracking** (in a video) is an image/video processing task that locates one or multiple moving objects over time. The goal of the task is to find an association between the target object(s) in the successive video frames. The task becomes difficult when the objects move faster relative to the frame rate or when the object to be tracked changes its orientation over time. The object tracking systems use a motion model taking into account how the target object may change for different possible motions of the object.

Object tracking is useful in human-computer interaction, security/surveillance, traffic control, and many more areas. Since it considers the appearance and the location of an object in the past frame, under certain circumstances, we may still be able to track an object despite the object detection fails. Few tracking algorithms that perform local searches are very fast. Hence, it's generally a good strategy to track an object indefinitely once it is detected for the first time. Most real-world applications implement tracking and detection simultaneously.

In this recipe, you will learn how to track multiple objects in a video using ``opencv-python`` functions, where the object locations in the very first frame will be provided to you in terms of bounding box coordinates.

### How it works...
The **MultiTracker** class from OpenCV was used to implement multi-object tracking. The multi-object tracker (which is implemented simply as a collection of single-object trackers) processes the tracked objects independently.

A multi-object tracker needs two inputs, namely, a reference video frame (we used the first video frame as reference) and locations (to be specified in terms of bounding boxes) of all of the objects (in the reference frame) that we want to track. Then, the tracker simultaneously tracks the locations of the target objects in the succeeding frames.

OpenCV has eight different object trackers (types): BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, and CSRT. The **KCF** tracker is fast and accurate, the **CSRT** tracker is more accurate than KCF but slower, whereas the **MOSSE** tracker is extremely fast but not as accurate as either KCF or CSRT.

In this recipe, the CSRT tracker was used; its implementation is based on a discriminative correlation filter with spatial and channel reliability.

#### issue
* https://github.com/opencv/opencv-python/issues/441
```python
Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> cv2.legacy.MultiTracker_create()
<legacy_MultiTracker 00000203D312A9B0>
```
But in windows , it does not work !

### There's more...
Track multiple objects using the KCF, MOSSE, and GOTURN trackers (the last one is a deep learning-based object tracker), and compare the results obtained with the CSRT tracker.

### See also
Refer to the following links to learn more about this recipe:
* https://en.wikipedia.org/wiki/Video_tracking
* https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
* https://docs.opencv.org/3.4/d9/df8/group__tracking.html
* https://www.youtube.com/watch?v=GPTlMZQ6f8o

## Text detection/recognition in images with EAST/Tesseract
First, download the  model from https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb , and put it in the appropriate path inside the models folder. 
```bash
wget https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb 
```
tutorials: 
* https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
* https://www.kaggle.com/datasets/yelmurat/frozen-east-text-detection
If OS is Ubuntu or Colab:
```
!sudo apt update 
!sudo apt install -y  tesseract-ocr  
```

First, download the pretrained EAST text detector model from https://codeload.github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/zip and extract the compressed model in the appropriate path inside the models folder. Install Tesseract (v4) from http://emop.tamu.edu/Installing-Tesseract-Windows8, for example, and the ``pytesseract`` package with pip. Import the required libraries using the following code snippet:
```python
import pytesseract
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np
```

### How it works...
Running a forward pass on the pretrained EAST model returns scores and geometry that are decoded using the decode_predictions() function to obtain the bounding boxes (ROIs) predicted to be containing text. Next, the texts inside these ROIs are to be extracted with the ``pytesseract image_to_string()`` method.

To extract texts as strings using Tesseract v4 OCR, the command-line arguments are needed to be passed as a configuration to the ``image_to_string()`` method of ``pytesseract`` (for example, we used "``-l eng --oem 1 --psm 11``" as the configuration):
* A language (English, configuration)
* An OEM flag=1 (use an **LSTM(Long Short-Term Memory)** model for OCR)
* An OEM value=11 (treat as sparse text, that is, find as much text as possible in no particular order)

### See also
Refer to the following links to learn more about this recipe:
* https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
* https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
* https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/
* https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
* https://arxiv.org/pdf/1704.03155.pdf
* https://pdfs.semanticscholar.org/d933/a6d0049f53344c5384c0905afe463a086bdb.pdf?_ga=2.137491379.201554747.1577481675-754696371.1577481675

## Face detection with Viola-Jones/Haar-like features 
```bash
!wget https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/opencv_face_detector.pbtxt
!wget https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb
```
Haar-like features are very useful image features used in object detection. They were introduced in the very first real-time face detector by **Viola** and **Jones**. Using integral images, **Haar-like features** of any size (scale) can be efficiently computed in constant time. The computation speed is the key advantage of a **Haar-like feature** over most other features. Using the **Viola-Jones** face detection algorithm, faces can be detected in an image using these **Haar-like features**. Each Haar-like feature acts as just a weak classifier, and hence a huge number of these features are required to detect a face with good accuracy. Therefore, a large number of features are computed for all possible locations and sizes of each Haar-like kernel, using the integral images. Then, an **AdaBoost** ensemble classifier is used to select important features from the huge number of features and combine them into a strong classifier model during the training phase. The model learned is then used to classify a face region with the selected features and can be used as a face detector.

In this recipe, you will learn how to use OpenCV's pretrained classifiers (that is, detectors) for face and eyes, to detect human faces in an image. These pretrained classifiers are serialized as XML files and come with an OpenCV installation (this can be found in the 'opencv/data/haarcascades/' folder). You may need to download the pretrained classifier for smile detection—if it's not already there, it can be found here: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml.

### How it works...
The faces in the image can be found using the cv2.detectMultiScale() function, with the pretrained face cascade classifier. This function accepts the following parameters: 
* ``scaleFactor``: This is a parameter that specifies the extent to which the image size is decreased for each image scale and used to create a scale pyramid (for example, a scale factor of 1.2 means reducing the size by 20%). The smaller the scaleFactor  parameter, the higher the chance that a matching size is found (for detection, with the model).
* ``minNeighbors``: This is a parameter that specifies the number of neighbors each candidate rectangle needs to keep. This parameter affects the quality of the detected faces—a larger value enables detection with higher quality, but smaller in numbers.
* ``minSize`` and ``maxSize``: These are the minimum and maximum possible object sizes, respectively. Objects of sizes beyond these values will be ignored.

When faces are detected, the positions the faces are returned by the function as a list of ``Rect(x, y, w,h)``.

Once a face bounding box is obtained, it defines the ROI for the face, and then the eye/smile detection on this ROI can be applied (since the eyes/smile are always to be found on the face).
### There's more...
Use the dlib HOG-based frontal face detector and OpenCV's Single Shot MuliBox Detector (SSD) pretrained deep learning model to detect faces in images. Compare different face detectors' performances (in terms of accuracy and time complexity). Try with face images with different angles/orientations and with eye glasses. Detect eyes with the dlib facial landmark detection.

### See also
Refer to the following links to learn more about this recipe:
* https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-IJCV-01.pdf
* https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
* https://arxiv.org/abs/1512.02325
* https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/
* https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml
* https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#a90fe1b7778bed4a27aa8482e1eecc116
Chapter 7 of the book, Hands-on Image Processing with Python

# Using Docker
## Building Dockerfiles

The image is about 7.5 GB.
```bash
docker build -t jupyter .
```

You can add mount parameters to the run command with the -v option. This is useful if you want the docker container to share directories with the host machine. Otherwise, you can copy files using the docker-copy command
```bash
docker run -it --rm --net=host jupyter
```

## Jupyter Docker Stacks
* https://jupyter-docker-stacks.readthedocs.io/en/latest/
* https://github.com/jupyter/docker-stacks/tree/main/images/scipy-notebook
```
docker run -p 10000:8888 quay.io/jupyter/scipy-notebook:2024-01-15

docker run -it --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work quay.io/jupyter/datascience-notebook:2024-01-15
```

http://10.100.198.102:10000/

We use old image: https://jupyter-docker-stacks.readthedocs.io/en/latest/#using-old-images
* https://github.com/jupyter/docker-stacks/tree/main/images/datascience-notebook
``` 
docker run -it --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work quay.io/jupyter/datascience-notebook:b86753318aa1
```

http://10.100.198.102:10000/

# Podman Machine Cli
## for winodws
* Install windows wsl2
```powershell
wsl --install --no-distribution
```
* Install podman-cli
```powershell
choco install -y podman-cli
```
# Recipes provide ways to be more productive in Jupyter and Python
* Obtaining the history of Jupyter commands and outputs
* Auto-reloading packages
* Debugging
  * Timing code execution
  * Displaying progress bars
* Compiling your code
* Speeding up pandas DataFrames
* Parallelizing your code

## Obtaining the history of Jupyter commands and outputs
There are lots of different ways to obtain the code in Jupyter cells programmatically. Apart from these inputs, you can also look at the generated outputs. We'll get to both, and we can use global variables for this purpose.

### Execution history
In order to get the execution history of your cells, the ``_ih`` list holds the code of executed cells. In order to get the complete execution history and write it to a file, you can do the following:
```python
with open('command_history.py', 'w') as file:
    for cell_input in _ih[:-1]:
        file.write(cell_input + '\n')
```

If up to this point, we only ran a single cell consisting of ``print('hello, world!')``, that's exactly what we should see in our newly created file, ``command_history.py``:
```python
!cat command_history.py
print('hello, world!')
```
On Windows, to print the content of a file, you can use the ``type`` command.

Instead of ``_ih``, we can use a shorthand for the content of the last three cells.``_i`` gives you the code of the cell that just executed, ``_ii`` is used for the code of the cell executed before that, and ``_iii`` for the one before that.

### Outputs
In order to get recent outputs, you can use ``_`` (single underscore), ``__`` (double underscore), and ``___`` (triple underscore), respectively, for the most recent, second, and third most recent outputs.
## Auto-reloading packages
``autoreload`` is a built-in extension that reloads the module when you make changes to a module on disk. It will automagically reload the module once you've saved it. 

Instead of manually reloading your package or restarting the notebook, with ``autoreload``, the only thing you have to do is to load and enable the extension, and it will do its magic.

We first load the extension as follows:
```python 
%load_ext autoreload
```
And then we enable it as follows:
```python
%autoreload 2
```

This can save a lot of time when you are developing (and testing) a library or module. 
## Debugging
If you cannot spot an error and the traceback of the error is not enough to find the problem, debugging can speed up the error-searching process a lot. Let's have a quick look at the debug magic:
1. Put the following code into a cell:
   ```python
   def normalize(x, norm=10.0):
     return x / norm

   normalize(5, 1)
   ```
   You should see 5.0 as the cell output.

   However, there's an error in the function, and I am sure the attentive reader will already have spotted it. Let's debug!
2. Put this into a new cell:
   ```python
   %debug
   normalize(5, 0)
   ```
3. Execute the cell by pressing ``Ctrl + Enter`` or ``Alt + Enter``. You will get a debug prompt:
   ```python
   > <iPython-input-11-a940a356f993>(2)normalize() 
        1 def normalize(x, norm=10): ----> 
        2   return x / norm 
        3 
        4 normalize(5, 1) 
   ipdb> a 
   x = 5 
   norm = 0 
   ipdb> q
   --------------------------------------------------------------------------- ZeroDivisionError                         Traceback (most recent call last)
   <iPython-input-13-8ade44ebcb0c> in <module>()
        1 get_iPython().magic('debug') ---->
        2 normalize(5, 0)


   <iPython-input-11-a940a356f993> in normalize(a, norm)
        1 def normalize(x, norm=10): ----> 
        2   return x / norm 
        3 
        4 normalize(5, 1) 
   ZeroDivisionError: division by zero
   ```
   We've used the argument command to print out the arguments of the executed function, and then we quit the debugger with the quit command. You can find more commands on **The Python Debugger (pdb)** documentation page at https://docs.Python.org/3/library/pdb.html.

   Let's look at a few more useful magic commands. 
### Timing code execution
Once your code does what it's supposed to, you often get into squeezing every bit of performance out of your models or algorithms. For this, you'll check execution times and create benchmarks using them. Let's see how to time executions.

There is a built-in magic command for timing cell execution – ``timeit``. The ``timeit`` functionality is part of the Python standard library (https://docs.Python.org/3/library/timeit.html). It runs a command 10,000 times (by default) in a period of 5 times inside a loop (by default) and shows an average execution time as a result:
```python
%%timeit -n 10 -r 1
import time
time.sleep(1)
```
We see the following output:
```python
1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 10 loops each)
```
The ``iPython-autotime`` library (https://github.com/cpcloud/iPython-autotime) is an external extension that provides you the  timings for all the cells that execute, rather than having to use %%timeit every time:

1. Install ``autotime`` as follows:
   ```python
   pip install iPython-autotime
   ```
   Please note that this syntax works for Colab, but not in standard Jupyter Notebook. What always works to install libraries is using the pip or conda magic commands, ``%pip`` and ``%conda``, respectively. Also, you can  execute any  shell command from the notebook if you start your line with an exclamation mark, like this:
   ```python
   !pip install iPython-autotime
   ```
2. Now let's use it, as follows:
   ```python
   %load_ext autotime
   ```
   Test how long a simple list comprehension takes with the following command:
   ```python
   sum([i for i in range(10)])
   ```
   We'll see this output: 
   ```bash
   time: 5.62 ms.
   ```

Hopefully, you can see how this can come in handy for comparing different implementations. Especially in situations where you have a lot of data, or complex processing, this can be very useful.
### Displaying progress bars
Even if your code is optimized, it's good to know if it's going to finish in minutes, hours, or days. ``tqdm`` provides progress bars with time estimates. If you aren't sure how long your job will run, it's just one letter away – in many cases, it's just a matter of changing ``range`` for ``trange``:
   ```python
   from tqdm.notebook import trange
   from tqdm.notebook import tqdm
   tqdm.pandas()
   ```
The ``tqdm`` pandas integration (optional) means that you can see progress bars for pandas ``apply`` operations. Just swap ``apply`` for ``progress_apply``.

For Python loops just wrap your loop with a tqdm function and voila, there'll be a progress bar and time estimates for your loop completion!
```python
global_sum = 0.0
for i in trange(1000000):
   global_sum += 1.0
```
Tqdm provides different ways to do this, and they all require minimal code changes - sometimes as little as one letter, as you can see in the previous example. The more general syntax is wrapping your loop iterator with tqdm like this:
```python
for _ in tqdm(range(10)):
   print()
```
So, next time you are just about to set off long-running loop, and you are not just sure how long it will take, just remember this sub-recipe, and use ``tqdm``.

## Compiling your code
Python is an interpreted language, which is a great advantage for experimenting, but it can be detrimental to speed. There are different ways to compile your Python code, or to use compiled code from Python.

Let's first look at Cython. Cython is an optimizing static compiler for Python, and the programming language compiled by the Cython compiler. The main idea is to write code in a language very similar to Python, and generate C code. This C code can then be compiled as a binary Python extension. SciPy (and NumPy), scikit-learn, and many other libraries have significant parts written in Cython for speed up. You can find out more about Cython on its website at https://cython.org/:

1. You can use the Cython extension for building cython functions in your notebook:
   ```python
   %load_ext Cython
   ```
2. After loading the extension, annotate your cell as follows:
   ```python
   %%cython
   def multiply(float x, float y):
       return x * y
   ```
3. We can call this function just like any Python function – with the added benefit that it's already compiled:
   ```python
   multiply(10, 5)  # 50
   ```
   This is perhaps not the most useful example of compiling code. For such a small function, the overhead of compilation is too big. You would probably want to compile something that's a bit more complex. 

   Numba is a JIT compiler for Python (https://numba.pydata.org/). You can often get a speed-up similar to C or Cython using numba and writing idiomatic Python code like the following:
   ```python
   from numba import jit
   @jit
   def add_numbers(N):
       a = 0
       for i in range(N):
           a += i
   add_numbers(10)           
   ```
   With autotime activated, you should see something like this: 
   ```python
   time: 2.19 s          
   ```
   So again, the overhead of the compilation is too big to make a meaningful impact. Of course, we'd only see the benefit if it's offset against the compilation. However, if we use this function again, we should see a speedup. Try it out yourself! Once the code is already compiled, the time significantly improves:
   ```python
   add_numbers(10)      
   ```
   You should see something like this:
   ```python
   time: 867 µs    
   ```
   There are other libraries that provide JIT compilation including TensorFlow, PyTorch, and JAX, that can help you get similar benefits.

   The following example comes directly from the JAX documentation, at https://jax.readthedocs.io/en/latest/index.html:
   ```python
   import jax.numpy as np
   from jax import jit
   def slow_f(x):
       return x * x + x * 2.0

   x = np.ones((5000, 5000)) 
   fast_f = jit(slow_f) 
   fast_f(x)    
   ```
So there are different ways to get speed benefits from using JIT or ahead-of-time compilation. We'll see some other ways of speeding up your code in the following sections.
## Speeding up pandas DataFrames
One of the most important libraries throughout this book will be ``pandas``, a library for tabular data that's useful for **Extract, Transform, Load (ETL)** jobs. Pandas is a wonderful library, however; once you get to more demanding tasks, you'll hit some of its limitations. Pandas is the go-to library for loading and transforming data. One problem with data processing is that it can be slow, even if you vectorize the function or if you use ``df.apply()``.

You can move further by parallelizing ``apply``. Some libraries, such as ``swifter``, can help you by choosing backends for computations for you, or you can make the choice yourself:

* You can use Dask DataFrames instead of pandas if you want to run on multiple cores of the same or several machines over a network.
* You can use CuPy or cuDF if you want to run computations on the GPU instead of the CPU. These have stable integrations with Dask, so you can run both on multiple cores and multiple GPUs, and you can still rely on a pandas-like syntax (see https://docs.dask.org/en/latest/gpu.html).

As we've mentioned, ``swifter`` can choose a backend for you with no change of syntax. Here is a quick setup for using ``pandas`` with ``swifter``:
```python
mport pandas as pd
import swifter

df = pd.read_csv('some_big_dataset.csv')
df['datacol'] = df['datacol'].swifter.apply(some_long_running_function)
```
Generally, apply() is much faster than looping over DataFrames.

You can further improve the speed of execution by using the underlying NumPy arrays directly and accessing NumPy functions, for example, using ``df.values.apply()``. NumPy vectorization can be a breeze, really. See the following example of applying a NumPy vectorization on a pandas DataFrame column:
```python
squarer = lambda t: t ** 2
vfunc = np.vectorize(squarer)
df['squared'] = vfunc(df[col].values)
```
These are just two ways, but if you look at the next sub-recipe, you should be able to write a parallel map function as yet another alternative.
## Parallelizing your code
One way to get something done more quickly is to do multiple things at once. There are different ways to implement your routines or algorithms with parallelism. Python has a lot of libraries that support this functionality. Let's see a few examples with multiprocessing, Ray, joblib, and how to make use of scikit-learn's parallelism.

The multiprocessing library comes as part of Python's standard library. Let's look at it first. We don't provide a dataset of millions of points here – the point is to show a usage pattern – however, please imagine a large dataset. Here's a code snippet of using our pseudo-dataset:
```python
# run on multiple cores
import multiprocessing

dataset = [
    {
        'data': 'large arrays and pandas DataFrames',
        'filename': 'path/to/files/image_1.png'
    }, # ... 100,000 datapoints
]

def get_filename(datapoint):
    return datapoint['filename'].split('/')[-1]

pool = multiprocessing.Pool(64)
result = pool.map(get_filename, dataset)
```
Using Ray, you can parallelize over multiple machines in addition to multiple cores, leaving your code virtually unchanged. Ray efficiently handles data through shared memory (and zero-copy serialization) and uses a distributed task scheduler with fault tolerance:
```python
# run on multiple machines and their cores
import ray
ray.init(ignore_reinit_error=True)

@ray.remote
def get_filename(datapoint):
    return datapoint['filename'].split('/')[-1]

result = []
for datapoint in dataset:
    result.append(get_filename.remote(datapoint))
```
Scikit-learn, the machine learning library we installed earlier, internally uses joblib for parallelization. The following is an example of this:

```python
from joblib import Parallel, delayed

def complex_function(x):
    '''this is an example for a function that potentially coult take very long.
    '''
    return sqrt(x)

Parallel(n_jobs=2)(delayed(complex_function)(i ** 2) for i in range(10))
```
This would give you [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]. We took this example from the joblib examples about parallel for loops, available at https://joblib.readthedocs.io/en/latest/parallel.html.

When using scikit-learn, watch out for functions that have an n_jobs parameter. This parameter is directly handed over to joblib.Parallel (https://github.com/joblib/joblib/blob/master/joblib/parallel.py). none (the default setting) means sequential execution, in other words, no parallelism. So if you want to execute code in parallel, make sure to set this n_jobs parameter, for example, to -1 in order to make full use of all your CPUs.

PyTorch and Keras both support multi-GPU and multi-CPU execution. Multi-core parallelization is done by default. Multi-machine execution in Keras is getting easier from release to release with TensorFlow as the default backend. 

## See also
While notebooks are convenient, they are often messy, not conducive to good coding habits, and they cannot be versioned cleanly. Fastai has developed an extension for literate code development in notebooks called nbdev (https://github.com/fastai/nbdev), which provides tools for exporting and documenting code.

There are a lot more useful extensions that you can find in different places:

* The extension index: https://github.com/iPython/iPython/wiki/Extensions-Index
* Jupyter contrib extensions: https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions.html
* The awesome-jupyter list: https://github.com/markusschanta/awesome-jupyter

We would also like to highlight the following extensions:
* SQL Magic, which performs SQL queries: https://github.com/catherinedevlin/iPython-sql
* Watermark, which extracts version information for used packages: https://github.com/rasbt/watermark
* Pyheatmagic, for profiling with heat maps: https://github.com/csurfer/pyheatmagic
* Nose testing, for testing using nose: https://github.com/taavi/iPython_nose
* Pytest magic, for testing using pytest: https://github.com/cjdrake/iPython-magic
* Dot and others, used for drawing diagrams using graphviz: https://github.com/cjdrake/iPython-magic
* Scalene, a CPU and memory profiler: https://github.com/emeryberger/scalene

Some other libraries used or mentioned in this recipe include the following:
* Swifter: https://github.com/jmcarpenter2/swifter
* Autoreload: https://iPython.org/iPython-doc/3/config/extensions/autoreload.html
* pdb: https://docs.Python.org/3/library/pdb.html
* tqdm: https://github.com/tqdm/tqdm
* JAX: https://jax.readthedocs.io/
* Seaborn: https://seaborn.pydata.org/
* Numba: https://numba.pydata.org/numba-doc/latest/index.html
* Dask: https://ml.dask.org/
* CuPy: https://cupy.chainer.org
* cuDF: https://github.com/rapidsai/cudf
* Ray: http://ray.readthedocs.io/en/latest/rllib.html
* joblib: https://joblib.readthedocs.io/en/latest/
* Classifying in scikit-lea