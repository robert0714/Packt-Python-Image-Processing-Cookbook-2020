# Face Recognition, Image Captioning, and More
In this chapter, we will discuss the application of a few advanced machine learning and deep learning techniques to solve a few advanced image processing problems. We will start with a face recognition problem that tries to match a set of faces detected in an image with a fixed set of known faces using deep face embedding representations. You will then learn how to use a few deep learning models to solve problems, such as the age or gender recognition of a human face and automatically colorizing a grayscale image. Another interesting problem we will look at is automatically captioning images with a deep learning model called **im2txt**. Finally, we will concentrate on a few techniques for image generation. In particular, we will focus on **generative models** in image processing (for example, a GAN, a VAE, and an RBM), which is a hot topic in image processing. The term **generative models** (often contrasted with **discriminative models**, such as SVMs/logistic regression) refers to the class of machine learning/deep learning models that tries to model the generation or distribution of input data (for example, images) by learning a probabilistic model. The goal is to generate new data (images) by sampling from the model learned.

In this chapter, you will learn about the following recipes:

1. Face recognition using FaceNet (a deep learning model)
1. Age, gender, and emotion recognition using deep learning models
1. Image colorization with deep learning
1. Automatic image captioning with a CNN and an LSTM
1. Image generation with a GAN
1. Using a variational autoencoder to reconstruct and generate images
1. Using a restricted Boltzmann machine to reconstruct Bangla MNIST images
* colab: https://colab.research.google.com/github/robert0714/Packt-Python-Image-Processing-Cookbook-2020/blob/main/Chapter%2009/Chapter%2009.ipynb

## Face recognition using FaceNet
**Face recognition** is an image processing/computer vision task that tries to identify and verify a person based on an image of their face. Face recognition problems can be categorized into two different types:

* **Face verification** (*is this the claimed person?*): This is a 1:1 matching problem (for example, a mobile phone that unlocks using a specific face uses face verification).
* **Face identification** (*who is this person?*): This is a 1:K matching problem (for example, an employee entering an office can be identified by face identification).

**FaceNet** is a unified system for face recognition (for both verification and identification). It is sometimes called a **Siamese network**. It is based on learning a Euclidean embedding per image using a deep convolutional network that encodes an image of a face into a vector of 128 numbers. The network is trained (via a **triplet loss function**) in a way that the square of the **L2** distances in the embedding space directly relates to the facial similarities. In this recipe, we will use a set of face images of six mathematicians—Bayes, Erdos, Euler, Gauss, Markov, and Turing (around 12 images for each of them in the training dataset and around 6 images for each of them in the test dataset). Although we are not going to train you in how to use FaceNet, the following diagram shows you how the system works (if it was trained from scratch with the **triplet loss function**). FaceNet learns an embedding f(x), where x is an input image. The model learns the parameters in such a way that the ||f(x(i))- f(x(j))||2 L2 norm is small when x(i) and x(j) are faces of the same person (positive) and large when the faces correspond to different people (negative), where f(.) represents the encoding (embedding) function presented by the deep CNN, as in the https://arxiv.org/pdf/1503.03832.pdf diagram.

In this recipe, you will learn how to use a pre-trained FaceNet model (in Keras) for face identification in order to identify a given face in an image as one of the six mathematicians' faces. You will convert the face recognition problem into a multi-class classification problem in the embedding space.
### Getting ready
First, download the pre-trained FaceNet model from https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_. Then, extract the model to the models folder. Download the images of the six mathematicians from the internet (you can use an automated script from https://github.com/hardikvasa/google-images-download). Download 20 images for each mathematician and for each mathematician, put 12 images in the train folder and the other 8 images in the test folder, as in the following screenshot (the test folder will also contain sub-folders with the mathematicians' names):

```bash
tree images -d

images
|-- anime
|   `-- images
|-- captioning
|-- mathematicians
|   |-- test
|   |   |-- Bayes+Thomas
|   |   |-- Erdos
|   |   |-- Euler
|   |   |-- Gauss
|   |   |-- Markov+Andrey
|   |   `-- Turing
|   `-- train
|       |-- Bayes+Thomas
|       |-- Erdos
|       |-- Euler
|       |-- Gauss
|       |-- Markov+Andrey
|       `-- Turing
|-- musicians
`-- tocolorize

!pip install mtcnn  scikit-learn
!wget https://github.com/a-m-k-18/Face-Recognition-System/raw/master/facenet_keras.h5 -P models/
!wget https://drive.google.com/file/d/1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1/view?usp=drive_link
!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1' -O facenet_keras.h5
```
### Issue 1: tensorflow load data: bad marshal data
* https://stackoverflow.com/questions/63484172/tensorflow-load-data-bad-marshal-data
* https://stackoverflow.com/questions/67653618/unable-to-load-facenet-keras-h5-model-in-python
* https://stackoverflow.com/questions/74556149/load-facenet-model
```bash
pip install keras-facenet keras-resnet
```
Old method
```python
from tensorflow.keras.models import load_model
model = load_model('models/facenet_keras.h5')
```
New method
```python
from keras_facenet import FaceNet
model = FaceNet()
```
### Issue 2: 'Model' object has no attribute 'predict'
* https://stackoverflow.com/questions/65647833/resnet-object-has-no-attribute-predict
* https://stackoverflow.com/questions/44806125/attributeerror-model-object-has-no-attribute-predict-classes
* https://stackoverflow.com/questions/69771967/moduleattributeerror-model-object-has-no-attribute-predict
* https://stackoverflow.com/questions/72602796/tensorflow-2-8-attributeerror-userobject-object-has-no-attribute-predict



## See also
For more details about this recipe, refer to the following links:

1. https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
1. https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
1. https://arxiv.org/pdf/1503.03832.pdf
1. https://sandipanweb.wordpress.com/2018/01/07/classifying-a-face-image-as-happy-unhappy-and-face-recognition-using-deep-learning-convolution-neural-net-with-keras-in-python/
1. https://www.youtube.com/watch?v=eHsErlPJWUU&list=PLD63A284B7615313A&index=14
1. https://www.youtube.com/watch?v=d2XB5-tuCWU
1. https://www.youtube.com/watch?v=EZmaYcdLfhM
1. https://www.thehindubusinessline.com/info-tech/google-backs-eus-proposed-facial-recognition-ban-microsoft-disagrees/article30616303.ece
1. https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf


## Age, gender, and emotion recognition using deep learning models
The age estimation of a face image can be posed as a deep classification problem using a CNN followed by an expected softmax value refinement (as can be done with a **Deep EXpectation (DEX)** model). In this recipe, you will first learn how to use a pre-trained deep learning model (a **WideResNet** with two classification layers added on top of it, which simultaneously estimates the age and gender using a single CNN) for age and gender recognition from a face image. We will use face images from the celebrity faces dataset for age and gender recognition. You will then implement emotion recognition using yet another pre-trained deep learning model, but this time you will need to detect the faces using a face detector (you could use transfer learning, too, and use your classifier on your own images, but this is left as an exercise for you to try).

### Getting ready
Download the pre-trained deep learning models from https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE and https://drive.google.com/file/d/0B6yZu81NrMhSV2ozYWZrenJXd1E. Extract the models to the appropriate paths in the models folder. Import the required libraries to start:
```python
import cv2
import dlib
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.models import model_from_json
from glob import glob
import matplotlib.pylab as plt
```
jupyter notebook
```
!wget https://github.com/jalajthanaki/Facial_emotion_recognition_using_Keras/raw/master/haarcascade_frontalface_alt2.xml -P models/
!wget https://github.com/jalajthanaki/Facial_emotion_recognition_using_Keras/raw/master/keras_model/model_5-49-0.62.hdf5 -P models/
```
### There's more...
Similarly, use the other pre-trained model to predict the emotion in a face in a given image (use the face detector function to detect the faces first). 

You may want to download the five celebrities' datasets from https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset and try different face images for emotion, gender, and age recognition.

### See also
For more details about this recipe, use the following links:
* https://github.com/yu4u/age-gender-estimation
* https://github.com/jalajthanaki/Facial_emotion_recognition_using_Keras
* https://stackoverflow.com/questions/53859419/dlib-get-frontal-face-detector-gets-faces-in-full-image-but-does-not-get-in-c
* https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf