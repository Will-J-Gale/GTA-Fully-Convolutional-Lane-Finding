# GTA-Fully-Convolutional-Lane-Finding
![alt text](https://github.com/Will-J-Gale/GTA-Fully-Convolutional-Lane-Finding/blob/master/Images/3.%20LaneFinding.gif)  

## Updates
__1. PNG Images__
 
 * The original training data used JPEG images, which turned out to have JPEG compression artefacts and affected the training, whereas    the new training data uses PNG format to avoid heavy compression artefacts  
  
__2. Threshold Training Data:__

 * The new training data images have a threshold applied to them, so the lanes are guaranteed to all be at 1 and background all be at 0, with nothing in between.  
  
__3. Classification:__

 * The original model posed lane finding as a regression problem, using a "mean squared error" loss with the output activation as "relu". However, the new problem poses the lane finding as a classification problem, and used a "binary cros entropy" loss with a sigmoid activation on the output. This is more in line with the SegNet, and gives much better results.   

Overall these changes drastically improve the lane finding.


## Model Progression

![alt text](https://github.com/Will-J-Gale/GTA-Fully-Convolutional-Lane-Finding/blob/master/Images/1.%20Original_Model.gif)
![alt text](https://github.com/Will-J-Gale/GTA-Fully-Convolutional-Lane-Finding/blob/master/Images/2.%20Cleaned_Data.gif)
![alt text](https://github.com/Will-J-Gale/GTA-Fully-Convolutional-Lane-Finding/blob/master/Images/3.%20LaneFinding.gif) 

These 3 images show the progression of the algorithm

__Top Left:__ Old model with JPEG training images that have artefacts, trained as a regression problem  
__Top Right:__ Trained with clean PNG images, trained as a regression problem  
__Bottom Left:__ Newest model with clean training data, trained as a classification problem  

It is clear that training the algorithm as a classification problem gave much better results. Moreover, clean training data also helped improve the model.

## How it works
This algorithm is a fully convolutional neural network that takes a 3-channel image input and outputs a 1 channel image with its prediction of where the lanes are  

It is based on the networks:
   * https://towardsdatascience.com/lane-detection-with-deep-learning-part-1-9e096f3320b7
   * http://mi.eng.cam.ac.uk/projects/segnet/

The network was trained on ~40000 images   
~20000 images were used but were flipped horizontally doubling the sample size  
Lanes were captured using https://github.com/Will-J-Gale/GTA-Lane-Finding with only the good images used

## Algorithm Evaluation
This algorithm seems to learn to find the lane is currently in. Unlike the previous iterations, it more precisely detects the edges of the lanes. It fails when there are no clear lines (at junctions) but with further training on a more robust data set, this problem could be fixed.

## Advantages over standard lane finding algorithm
   1. Runs at ~20fps (~10fps faster than standard algorithm)
   2. Reacts well to fast movement
   3. Potentially more accurate 
   
## Prerequisites 
1. GTA 5 + Mods (Not all mods are 100% necessary)
   * Script Hook V
   * Native trainer
   * Enhanced native trainer
   * GTA V FoV v1.35
   * Extended Camera Settings
   * Hood Camera 
2. Python 3.6
3. OpenCV
4. Numpy
5. Tensorflow GPU
6. Keras

## Usage
Recommended to use on dual monitors
1. Run GTA5 in windowed mode 1280x720
2. Find a car and enable Hood Camera
3. Run LaneFinder.py
   
The demonstration at the top of the page shows the fully convolution network has potential and is clearly learning the lanes.  
More data is still required to reduce the glitches and improve accuracy.

