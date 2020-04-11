
# Importing libraries

import numpy as np
import cv2
from tqdm import tqdm
import time
import os
from scipy.special import expit
import signal
import glob


# Initializing variables
min_angle = 0
max_angle = 0


# Image handling
def read_image(image):
    # Keeping only the red channel of image 
    im_full = image[:, :, 2]
    # Resizing image 
    im_full = cv2.resize(im_full, (60,60)) 
    # Return normalised image
    return np.array(im_full[30:,:])/255


# Train class
def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''
    global min_angle, max_angle
    X = []

    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    angles = data[:,1]
    theta = np.matrix(angles).transpose()
    
    # Calculating maximum and minimum steering angle values
    max_angle = 180
    min_angle = -180
       
    # Binning
    # Creating 64 bins of equal length between the maximum and minimum value of angles
    bins = np.linspace(min_angle,max_angle,64) 
    values = np.linspace(0,64 -1,64)
    y = np.zeros((len(theta),64))
        
    # Assigning angles to bins
    for i,angle in enumerate(theta):
        index =  int(np.interp(angle,bins, values))
        y[i,index] = 1
        
        # Creating a Gaussian distribution with a center on the target index 
        # Example of Gaussian Distribution [0 0.3 0.6 0.89 1 0.89 0.6 0.3 0]
        if index - 1 >= 0 and index + 1 < 64:
            y[i,index - 1] = 0.89
            y[i,index + 1] = 0.89
            
            if index - 2 >= 0 and  index + 2 < 64:
                y[i,index - 2] = 0.6
                y[i,index + 2] = 0.6
                
                if index - 3 >= 0 and  index + 3 < 64:
                    y[i,index - 2] = 0.3
                    y[i,index + 2] = 0.3
                    
    # Reading Files   
    paths = glob.glob(path_to_images+'/*.jpg')
    print("Reading Files")
    print("")
    
    for i,frame in tqdm(enumerate(frame_nums)):     
        im_full = cv2.imread(path_to_images + '/' + str(int(frame)).zfill(4) + '.jpg')
        im_full = im_full[:, :, 2]
        
        # Resize the image
        im_full = cv2.resize(im_full, (60,60)) 
        
        # Return normalised image
        X.append(np.array(im_full[30:,:])/255)
    
    # Hyperparameters
    iterations = 3300
    learning_rate = 1e-1*(4)
    
    # Creating an instance of the Neural Network Class
    NN = Neural_Network() 

    X = np.reshape(X,(1500,(1800)))
    print("Training the model")
    print("")
    
    loss = []
    
    # Training the model
    for iter in tqdm(range(iterations)):    
        # Compute Gradients
        gradients  = NN.computeGradients(X, y)
        # Get the weights
        params = NN.getParams()   
        # Perform gradient descent
        params[:] = params[:] - ((learning_rate*gradients[:])/(len(X)))  
        # Updating weights
        NN.setParams(params)
        loss.append(NN.costFunction(X, y))
    return NN


# Predict function
def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    # Creating Bins
    bins = np.linspace(min_angle,max_angle,64) 
    im_full = cv2.imread(image_file)
    im_full = im_full[:, :, 2]
    # Resizing image
    im_full = cv2.resize(im_full, (60,60))
    # Croping image
    image_vector = np.array(im_full[30:,:])/255
    # Flatting image
    image_vector = np.reshape(image_vector,(1,-1))
    # Return normalised image
    return bins[np.argmax(NN.forward(image_vector))]



class Neural_Network(object):
    def __init__(self):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 1800
        self.outputLayerSize = 64
        self.hiddenLayerSize = 128
        
        # Inititalize the weights 
        # Glorot Initialization: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf 
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)*np.sqrt(1/(self.inputLayerSize + self.hiddenLayerSize))
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)*np.sqrt(1/(self.outputLayerSize + self.hiddenLayerSize))
        
    def forward(self, X):
        # Propogate inputs though network 
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix 
        return expit(z)
    
    def sigmoidPrime(self,z):
        # Gradient of sigmoid 
        return (expit(z)*(1-expit(z)))
    
    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.multiply(np.dot(delta3, self.W2.T),self.sigmoidPrime(self.z2))
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
        
    # Helper Functions for interacting with other classes 
    def getParams(self):
        # Get W1 and W2 unrolled into vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
        
    def setParams(self, params):
        # Set W1 and W2 using single paramater vector
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
