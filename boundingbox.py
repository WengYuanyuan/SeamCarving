import cv2
import numpy as np


def findRect(mask):
    '''
    mask is a numpy array with n rows and m column, black backgound and white area,
    find the smallest n and m, biggist n and m
    '''
    #find w_min
    index = mask.argmax(axis=1)
    w_min = min(index[index!=0])
    
    #find h_min
    index = mask.argmax(axis=0)
    h_min = min(index[index!=0])
    
    #flip mask totally
    mask = mask[::-1, ::-1]

    #find w_max 
    index = mask.argmax(axis=1)
    w_max = min(index[index!=0])
    w_max = mask.shape[1] - w_max
    
    #find h_max
    index = mask.argmax(axis=0)
    h_max = min(index[index!=0])
    h_max = mask.shape[0] - h_max
    
    return h_min, w_min, h_max, w_max
    

if __name__ == "__main__":
    #set1 mask removal
    mask_remove = cv2.imread('mask_r.png',0)
    mask_remove = mask_remove[1:770,:]
    #mask_remove[0:99,:]=0
    #mask_remove[537:,:]=0
    #mask_remove[0:,:187]=0
    #mask_remove[:,901:]=0
    #mask_remove[mask_remove != 0]=255
    cv2.imwrite('mask_r2.png', mask_remove)
    mask_remove = cv2.imread('mask_r2.png',0)
    print np.unique(mask_remove)
    print mask_remove.shape

    h_min, w_min, h_max, w_max = findRect(mask_remove)
    print h_min, w_min, h_max, w_max
    