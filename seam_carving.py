# -*- coding: utf-8 -*-
"""
seam_carving.py: Seam carving for content-aware image resizing.
author: Yuanyuan Weng
email: yweng37@gatech.edu

"""
import cv2
import numpy as np
import sys
import timeit

# default index type
INDEX_DTYPE = np.uint32


def Simple_conv_EnergyImage(image):
    """
    energy funcion: SIMPLE
    input image: gray color image from original image
    Returns the energy map derived from gray color image
    """
    image_copyY=image.astype(np.int64)
    image_copyX=image.astype(np.int64)
    
    image_gradY=image_copyY[1:,:] - image_copyY[:-1,:]
    add = np.zeros((1,image_gradY.shape[1]))
    image_gradY = np.concatenate((add,image_gradY),axis=0)

    image_gradX=image_copyX[:,1:] - image_copyX[:,:-1]
    add = np.zeros((image_gradX.shape[0],1))
    image_gradX = np.concatenate((add,image_gradX),axis=1)
    energyMap = np.fabs(image_gradY)+np.fabs(image_gradX)

    return energyMap



def Sobel_conv_EnergyImage(image):
    """
        energy function: SOBEL
        input image: gray color image from original image
        Returns the energy map derived from gray color image
    """

    # based on the reference literature E(I)=|∂I/∂x|+|∂I/∂y|
    
    #the following implementation is wrong, cannot figure why
    #energyMap = np.fabs(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)) 
    #+ np.fabs(cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3))
    
    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    energyMap = np.fabs(x)+np.fabs(y)

    return energyMap

def findSingleSeam(energyMap):
    """
        input image: energy map derived from gray color image
        return a single vertical seam with lowest cost
    """

    energy_cum = np.zeros((energyMap.shape[0],energyMap.shape[1]+2),dtype=energyMap.dtype)
    energy_cum[:,0] = np.inf
    energy_cum[:,-1] = np.inf
    energy_cum[:,1:-1]=energyMap 
    pointer = np.zeros(energyMap.shape, dtype=np.uint8)


    # n represents row, and m represents column
    for n in range(1, energyMap.shape[0]):
        for m in range(energyMap.shape[1]):

            choices = energy_cum[n-1,m:m+3]
            smallest_m = choices.argmin()
            
            energy_cum[n,m+1] = energyMap[n,m] + choices[smallest_m]
            pointer[n,m] = smallest_m

    # extract n_seam seams
    extracted_seam = []
    column = energy_cum[-1,1:-1].argmin()
    width = energyMap.shape[0]
    for w in range(width-1, 0, -1):
        extracted_seam.append(column)
        column += pointer[w,column] - 1
    extracted_seam.append(column)

    return np.array(extracted_seam[::-1], dtype=INDEX_DTYPE)

def SeamDeduction(image, seam):
    """
        input image is in color, not gray
        return an image with one vertical seam removed
    """

    # the image_seamRemove has one less column than original image
    shape = list(image.shape)
    shape[1] -= 1
    image_seamRemove = np.zeros(shape, dtype=image.dtype)

    for i in range(image.shape[0]):
        image_seamRemove[i,0:seam[i]] = image[i,0:seam[i]]
        image_seamRemove[i,seam[i]:] = image[i,seam[i]+1:]

    return image_seamRemove

def seamInsertion(image, seams):
    """
        input: original image and series of seams extracted
        output: Enlarged image with insersion of a series of seams.
    """

    # create a new image
    n = image.shape[0]
    m = image.shape[1] + len(seams)
    l = image.shape[2]
    new = np.zeros(list((n,m,l)), dtype=image.dtype)

    # sort the seams by index, this will 
    seams.sort(axis=0)
    for i in range(n):
        current = seams[:,i]
        new[i,0:current[0]] = image[i,0:current[0]]
        for j in range(current.size):
            #vecterize the new image
            new[i,current[j-1]+j:current[j]+j] = image[i,current[j-1]:current[j]]
            #m
            new[i,current[j]+j] = image[i,current[j]]
        new[i,current[-1]+current.size:] = image[i,current[-1]:]
    
    return new

def seams_Marker(image, seams, mark_red=(255, 0, 0)):
    """
        input: image and inserted seams
        output: image with marking a series seams.
    """
    xi = np.arange(image.shape[0])
    # mark the image seam by seam
    for s in seams:
        image[xi,s] = mark_red

def resize(energy_function, img, new_resolution, mark=False):
    """
        parameters: 
            energy_function:simple or sobel 
            img:original image to work with
            new_resolution: target resolution in list [w,h]
        output the resized image
    """
    # calculate the time, start
    start = timeit.default_timer()

    # print out the message to start seam carving
    current = 0
    orig = sys.stdout
    sys.stdout = sys.stderr

    # determine resize parameters
    n = new_resolution[0]
    m = new_resolution[1]
    m0 = img.shape[1]
    n0 = img.shape[0]
    Snumbers = (n - m0, m - n0)
    total_seam_number = abs(Snumbers[0]) + abs(Snumbers[1])

    # mark the seams by creating a copy of original images
    if mark:
        marked_image = img.copy()

    # this process go through two iteration to process horizontal resize 
    # and then vertical resizing, but here for fig 5 and 8, there is only horizontal resizing
    # for imaging retargeting, using other file ordering.py to have a smarter order.
    # iteration 1: horizontal seams (vertical resize)
    for seam_number in Snumbers:
        remember_seams = mark or seam_number > 0 #if need to marker image or do seamInsertion, /
        #need to remember calculated seams
        image_original = img

        # maintain seams and an "original index" 2D array, if needed
        if remember_seams:
            horizontal_range = np.arange(n0)
            seams = np.zeros((0, n0), dtype=INDEX_DTYPE)
            image_range = np.arange(m0, dtype=INDEX_DTYPE)
            image_idx = np.tile(image_range, (n0, 1))
        
        # delete seam one by one
        for s in range(abs(seam_number)):                       
            #find one optimal vertical seam and delete it
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if energy_function == "simple":
                energyMap = Simple_conv_EnergyImage(gray_image)
            elif energy_function == "sobel":
                energyMap = Sobel_conv_EnergyImage(gray_image)  
            seam = findSingleSeam(energyMap)
            next_image = SeamDeduction(img, seam)

            
            if remember_seams:
                originalSeam = image_idx[horizontal_range, seam]
                image_idx = SeamDeduction(image_idx, seam)
                seams = np.vstack((seams, originalSeam))

            #update for next round
            img = next_image
            current += 1
            counter = "{0} of {1}".format(current, total_seam_number)
            print (counter),
            print("\b"*(len(counter)+2)),


        # using the remembered seams to mark original image
        if mark:
            img = image_original
            seams_Marker(marked_image, seams)
        elif seam_number > 0:
            img = seamInsertion(image_original, seams)

        # since findSingleSeam function can only find vertical seams, here we transpose 
        # the image
        img = img.transpose((1, 0, 2))
        if mark:
            marked_image = marked_image.transpose((1, 0, 2))


    print "done.{0}".format(" "*10)
    print "Time cost:",format(timeit.default_timer() - start)
    sys.stdout = orig
    
    if mark:
        return marked_image
    else:
        return img



if __name__ == "__main__":
    
    
    image = cv2.imread("bird.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    e = Simple_conv_EnergyImage(image)
    cv2.imwrite("energy.jpg", e)
    """ 
    for getting 50% width reduction on fig5.png
    the output is fig5_0.5.png
    reproduce fig5 in original paper
    """
    ## print "resize figure 5 by seam carving: "
    #img = cv2.imread("set2.jpg")
    #res = [640, 640]
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #energy_function = "sobel"
    #img_rsz = resize(energy_function, img_rgb, res, mark=False)
    #img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_RGB2BGR)
    #cv2.imwrite("set2_sc_noMarker.jpg", img_rsz)
    
    
    # """
    # for getting 50% enlargement on fig8.png
    # the output is fig8_1.5.png
    # reproduce fig8(c) in original paper
    # """
    # print "enlarge figure 8 by 50% with seam carving: "
    # img = cv2.imread("fig8.png")
    # res = [358,200]
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # energy_function = "sobel"
    # img_rsz = resize(energy_function, img_rgb, res, mark=False)
    # img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("fig8_1.5.png", img_rsz)
    #
    #
    # """
    # for getting 50% enlargement on fig8.png
    # and show the removed seams
    # the output is fig8_seam.png
    # reproduce fig8(d) in original paper
    # """
    # print "enlarge figure 8 by 50% and mark the seams with seam carving: "
    # img = cv2.imread("fig8_1.5.png")
    # res = [239,200]
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # energy_function = "sobel"
    # img_rsz = resize(energy_function, img_rgb, res, mark=True)
    # img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("fig8_seam.png", img_rsz)
    #
    #
    # """
    # for getting two-steps 50% enlargement on fig8.png
    # the output is fig8_2.0.png
    # reproduce fig8(f) in original paper
    # """
    # print "enlarge figure 8 by two steps of 50% with seam carving: "
    # img = cv2.imread("fig8_1.5.png")
    # res = [476,200]
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # energy_function = "sobel"
    # img_rsz = resize(energy_function, img_rgb, res, mark=False)
    # img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("fig8_2.0.png", img_rsz)
   
