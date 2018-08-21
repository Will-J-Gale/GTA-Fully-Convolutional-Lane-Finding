# GTA-Fully-Convolutional-Lane-Finding
![alt text](https://github.com/Will-J-Gale/GTA-Fully-Convolutional-Lane-Finding/blob/master/Images/Conv%20Lane%20HALF%20SIZE.gif)  

## How it works
This algorithm is a fully convolutional neural network that takes a 3 channel image input and outputs a 1 channel image with its prediction of where the lanes are  

It is based on the networks:
   * https://towardsdatascience.com/lane-detection-with-deep-learning-part-1-9e096f3320b7
   * http://mi.eng.cam.ac.uk/projects/segnet/

The network was trained on ~40000 images   
~20000 images were used but were flipped horizontally doubling the sample size  
Lanes were captured using https://github.com/Will-J-Gale/GTA-Lane-Finding with only the good images used

## Advantages over standard lane finding algorithm
   1. Runs at ~15fps (~5fps faster)
   2. Reacts well to fast movement
   3. Potentially more accurate 
   
## Prerequisites 
1. GTA 5 + Mods
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
   
The above demonstration shows the fully convolution network has potential and is clearly learning the lanes.  
More data is still required to reduce the glitches and improve accuracy

