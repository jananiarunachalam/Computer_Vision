# Autonomous Driving


## Objective
Train a neural network to steer a car using only image from a single camera. `neural_network.py` contains the basic functions and classes required to train the neural network. Method to train the neural network:

````
def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    
    Returns: 
    NN = Trained Neural Network object 
    '''
 ````

 Method returns a trained neural network class. Also, created a predict method:


````
def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given your trained neural network class, and an image filename, load image, make and return single predicted steering angle in degrees, as a float32. 
    '''
````

## Evalution 
The code is evaluated using `evaluate.py`. Performance will be evaluated by comparing your predicted steering angles to human steering angles.

## Packages
For this mini-project I have only used:
<ul>
<li>numpy
<li>opencv
<li>tdqm
<li>time
<li>scipy
</ul>

## The Data
Training data [here](http://www.welchlabs.io/unccv/autonomous_driving/data/training.zip). 
