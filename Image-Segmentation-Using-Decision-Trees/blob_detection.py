import numpy as np
# import sys
#It's ok to import whatever you want from the local util module if you would like:
# sys.path.append('../util')
# from data_handling import breakIntoGrids, reshapeIntoImage

def breakIntoGrids(im, s = 9):
    '''
    Break overall image into overlapping grids of size s x s, s must be odd.
    '''
    grids = []

    h = s//2 #half grid size minus one.
    for i in range(h, im.shape[0]-h):
        for j in range(h, im.shape[1]-h):
            grids.append(im[i-h:i+h+1,j-h:j+h+1].ravel())

    return np.vstack(grids)

def reshapeIntoImage(vector, im_shape, s = 9):
    '''
    Reshape vector back into image. 
    '''
    h = s//2 #half grid size minus one. 
    image = np.zeros(im_shape)
    image[h:-h, h:-h] = vector.reshape(im_shape[0]-2*h, im_shape[1]-2*h)

    return image


def count_fingers(im):
    '''
    Example submission for coding challenge. 
    
    Args: im (nxm) unsigned 8-bit grayscale image 
    Returns: One of three integers: 1, 2, 3
    
    '''

    ## ------ Input Pi[eline Develped in this Module ----- ##
    #You may use the finger pixel detection pipeline we developed in this module:
    #You may also replace this code with your own pipeline if you prefer
    im = im > 92 #Threshold image
    X = breakIntoGrids(im, s = 9) #Break into 9x9 grids

    #Use rule we learned with decision tree
    treeRule1 = lambda X: np.logical_and(np.logical_and(X[:, 40] == 1, X[:,0] == 0), X[:, 53] == 0)
    yhat = treeRule1(X)


    #Reshape prediction ino image:
    yhat_reshaped = reshapeIntoImage(yhat, im.shape)
    
    yhat_label = np.zeros(yhat_reshaped.shape)
    l = 1
    for y in range(1, (yhat_reshaped.shape[0])-1):
        for x in range(1, (yhat_reshaped.shape[1])-1):
            if yhat_reshaped[y][x] == 1:
                label = max(map(max,(yhat_label[(y-1):(y+2), (x-1):(x+2)])))
                if label == 0:
                    yhat_label[y][x] = l
                    l += 1
                else:
                    yhat_label[y][x] = label
    
    size = np.unique(yhat_label)
    blobs = 0
    for i in range(1, int(max(size))+1):
        if (yhat_label==size[i]).sum() >20:
            blobs+=1
        if blobs>=3:
            break
    
    return blobs