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