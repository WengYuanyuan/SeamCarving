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
import skimage
import skimage.io

# default index type
INDEX_DTYPE = np.uint32


# def Simple_conv_EnergyImage(image):
#     """
#     energy funcion: SIMPLE
#     input image: gray color image from original image
#     Returns the energy map derived from gray color image
#     """
#     image_copyY=image.astype(np.int64)
#     image_copyX=image.astype(np.int64)
#
#     image_gradY=image_copyY[1:,:] - image_copyY[:-1,:]
#     add = np.zeros((1,image_gradY.shape[1]))
#     image_gradY = np.concatenate((add,image_gradY),axis=0)
#
#     image_gradX=image_copyX[:,1:] - image_copyX[:,:-1]
#     add = np.zeros((image_gradX.shape[0],1))
#     image_gradX = np.concatenate((add,image_gradX),axis=1)
#     energyMap = np.fabs(image_gradY)+np.fabs(image_gradX)
#
#     return energyMap

def GradientMap(img):
    """
    energy funcion: SIMPLE
    input image: gray color image from original image
    Returns the energy map derived from gray color image
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_copyY = img.astype(np.int64)
    image_copyX = img.astype(np.int64)

    image_gradY = image_copyY[1:, :] - image_copyY[:-1, :]
    add = np.zeros((1, image_gradY.shape[1]))
    image_gradY = np.concatenate((add, image_gradY), axis=0)

    image_gradX = image_copyX[:, 1:] - image_copyX[:, :-1]
    add = np.zeros((image_gradX.shape[0], 1))
    image_gradX = np.concatenate((add, image_gradX), axis=1)
    energyMap = np.fabs(image_gradY) + np.fabs(image_gradX)

    return energyMap

# # original FEnergyMap, findSingleSeam #############
# ###################################################
# # Calculate forward energy map
# def FEnergyMap(img):
#     h0, w0 = img.shape[:2]
#     #h0 = 466
#     #w0 = 700
#     FEM = np.zeros((h0, w0))
#     gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     energy = GradientMap(gray_image)
#
#     FEM[0, :] = energy[0, :]
#
#     for i in range(1, h0):
#         for j in range(0, w0):
#             if j == 0:
#                 FEM[i, j] =energy[i, j] + min(FEM[i-1, j], FEM[i-1, j+1])
#             elif j == w0-1:
#                 FEM[i, j] = energy[i, j] + min(FEM[i-1, j-1], FEM[i-1, j])
#             else:
#                 Diff1 = abs(img[i, j+1, 0] - img[i, j-1, 0]) + abs(img[i, j+1, 1] - img[i, j-1, 1]) + abs(img[i, j+1, 2] - img[i, j-1, 2])
#                 Diff2 = abs(img[i-1, j, 0] - img[i, j-1, 0]) + abs(img[i-1, j, 1] - img[i, j-1, 1]) + abs(img[i-1, j, 2] - img[i, j-1, 2])
#                 Diff3 = abs(img[i-1, j, 0] - img[i, j+1, 0]) + abs(img[i-1, j, 1] - img[i, j+1, 1]) + abs(img[i-1, j, 2] - img[i, j+1, 2])
#
#                 FEM[i, j] = min(FEM[i-1, j-1] + Diff1 + Diff2, FEM[i-1, j] + Diff1, FEM[i-1, j+1] + Diff1 + Diff3)
#
#     return FEM
#
# def findSingleSeam(energyMap):
#     """
#         rewrite done except seam=[]
#         input image: energy map derived from gray color image
#         return a single vertical seam with lowest cost
#     """
#
#     energy_cum = np.zeros((energyMap.shape[0],energyMap.shape[1]+2),dtype=energyMap.dtype)
#     energy_cum[:,0] = np.inf
#     energy_cum[:,-1] = np.inf
#     energy_cum[:,1:-1]=energyMap
#     pointer = np.zeros(energyMap.shape, dtype=np.uint8)
#
#
#     # n represents row, and m represents column
#     for n in range(1, energyMap.shape[0]):
#         for m in range(energyMap.shape[1]):
#
#             choices = energy_cum[n-1,m:m+3]
#             smallest_m = choices.argmin()
#
#             energy_cum[n,m+1] = energyMap[n,m] + choices[smallest_m]
#             pointer[n,m] = smallest_m
#
#     # extract n_seam seams
#     extracted_seam = []
#     column = energy_cum[-1,1:-1].argmin()
#     width = energyMap.shape[0]
#     for w in range(width-1, 0, -1):
#         extracted_seam.append(column)
#         column += pointer[w,column] - 1
#     extracted_seam.append(column)
#
#     return np.array(extracted_seam[::-1], dtype=INDEX_DTYPE)


# Modified findSingleSeam #############
###################################################


def findSingleSeam(energyMap,img):
    """
        rewrite done except seam=[]
        input image: energy map derived from gray color image
        return a single vertical seam with lowest cost
    """

    FEM = np.zeros((energyMap.shape[0],energyMap.shape[1]),dtype=energyMap.dtype)

    FEM[0, :] = energyMap[0, :]
    pointer = np.zeros(energyMap.shape, dtype=np.uint8)

    # i represents row, and j represents column
    for i in range(1, energyMap.shape[0]):
        for j in range(0, energyMap.shape[1]):
            if j == 0:
                FEM[i, j] =energyMap[i, j] + min(FEM[i-1, j], FEM[i-1, j+1])

                # update pointer: which pixel was select in previous row
                choices = np.asarray([FEM[i-1, j], FEM[i-1, j+1]])
                pointer[i, j] = choices.argmin()+1

            elif j == energyMap.shape[1]-1:
                FEM[i, j] = energyMap[i, j] + min(FEM[i-1, j-1], FEM[i-1, j])

                # update pointer: which pixel was select in previous row
                choices = np.asarray([FEM[i-1, j-1], FEM[i-1, j]])
                pointer[i, j] = choices.argmin()

            else:
                Diff1 = abs(img[i, j+1, 0] - img[i, j-1, 0]) + abs(img[i, j+1, 1] - img[i, j-1, 1]) + abs(img[i, j+1, 2] - img[i, j-1, 2])
                Diff2 = abs(img[i-1, j, 0] - img[i, j-1, 0]) + abs(img[i-1, j, 1] - img[i, j-1, 1]) + abs(img[i-1, j, 2] - img[i, j-1, 2])
                Diff3 = abs(img[i-1, j, 0] - img[i, j+1, 0]) + abs(img[i-1, j, 1] - img[i, j+1, 1]) + abs(img[i-1, j, 2] - img[i, j+1, 2])

                FEM[i, j] = min(FEM[i-1, j-1] + Diff1 + Diff2, FEM[i-1, j] + Diff1, FEM[i-1, j+1] + Diff1 + Diff3)

                # update pointer: which pixel was select in previous row
                choices = np.asarray([FEM[i-1, j-1] + Diff1 + Diff2, FEM[i-1, j] + Diff1, FEM[i-1, j+1] + Diff1 + Diff3])
                pointer[i, j] = choices.argmin()

        #print pointer[i,:]
        #print FEM[i,:]
    cv2.imwrite('fig5_FEnergy.png',FEM)

    # extract nth_seam seams
    extracted_seam = []
    column = FEM[-1,:].argmin()
    height = energyMap.shape[0]
    for h in range(height-1, 0, -1):
        extracted_seam.append(column)
        
        column += pointer[h,column] - 1
    extracted_seam.append(column)
    #print extracted_seam
    #print np.array(extracted_seam[::-1], dtype=INDEX_DTYPE)
    
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

def resize(img, new_resolution, mark=False):
    """
        parameters: 
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

            energyMap = GradientMap(img)
            seam = findSingleSeam(energyMap,img)
            print seam
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
    
    """ 
    for getting 50% width reduction on fig5.png
    the output is fig5_0.5.png
    reproduce fig5 in original paper
    """
    # print "resize figure 5 by seam carving: "
    img = cv2.imread("fig5.png")
    # img = skimage.io.imread("fig5.png")

    # energyMap = GradientMap(img)
    # cv2.imwrite('fig5_FEnergy.png',energyMap)

    res = [650, 466]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rsz = resize(img_rgb, res, mark=False)
    img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_RGB2BGR)
    cv2.imwrite("fig5_forward.jpg", img_rsz)
