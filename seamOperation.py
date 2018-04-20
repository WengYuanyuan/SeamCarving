# -*- coding: utf-8 -*-
import cv2
import numpy as np
import weave

import timeit
import matplotlib.pyplot as plt

import pylab
import matplotlib.pyplot as plt

# try:
#    from setuptools import setup
#    from setuptools import extension
# except ImportError:
#    from distutils.core import setup
#    from distutils.extension import extension



"""
calculate transport map and choice bit map while computing optimal
seam ordering when resizing a given image. 0 is vertical seam and 1 is
horizontal seam
"""
# default index type
INDEX_DTYPE = np.uint32


def GradientEnergyMap(image):
    """
    energy funcion: SIMPLE
    input image: color image from original image
    Returns the energy map derived from gray color image
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image_copyY = image.astype(np.int64)
    image_copyX = image.astype(np.int64)

    image_gradY = image_copyY[1:, :] - image_copyY[:-1, :]
    add = np.zeros((1, image_gradY.shape[1]))
    image_gradY = np.concatenate((add, image_gradY), axis=0)

    image_gradX = image_copyX[:, 1:] - image_copyX[:, :-1]
    add = np.zeros((image_gradX.shape[0], 1))
    image_gradX = np.concatenate((add, image_gradX), axis=1)
    energyMap = np.fabs(image_gradY) + np.fabs(image_gradX)

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
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    energyMap = np.fabs(x)+np.fabs(y)

    return energyMap
def findVerSeam(image, energy_Map):

    # construct cumulative energy map and back pointers
    energy_cum = np.zeros((energy_Map.shape[0], energy_Map.shape[1] + 2), dtype=energy_Map.dtype)
    energy_cum[:, 0] = np.inf
    energy_cum[:, -1] = np.inf
    energy_cum[:, 1:-1] = energy_Map
    pointer = np.zeros(energy_Map.shape, dtype=np.uint8)

    # slow method
        # n represents row, and m represents column
    for n in range(1, energy_Map.shape[0]):
        for m in range(energy_Map.shape[1]):
            choices = energy_cum[n - 1, m:m + 3]
            smallest_m = choices.argmin()

            energy_cum[n, m + 1] = energy_Map[n, m] + choices[smallest_m]
            pointer[n, m] = smallest_m

    # extract n_seam seams
    extracted_seam = []
    column = energy_cum[-1, 1:-1].argmin()
    width = energy_Map.shape[0]
    for w in range(width - 1, 0, -1):
        extracted_seam.append(column)
        column += pointer[w, column] - 1
    extracted_seam.append(column)

    return np.array(extracted_seam[::-1], dtype=INDEX_DTYPE)


def findHorSeam(image, energy_Map):

    # transpost the original image
    # energy_Map = energy_Map.transpose((1, 0, 2))
    energy_Map = np.transpose(energy_Map)



    energy_cum = np.zeros((energy_Map.shape[0], energy_Map.shape[1] + 2), dtype=energy_Map.dtype)
    energy_cum[:, 0] = np.inf
    energy_cum[:, -1] = np.inf
    energy_cum[:, 1:-1] = energy_Map
    pointer = np.zeros(energy_Map.shape, dtype=np.uint8)

    # slow method
        # n represents row, and m represents column
    for n in range(1, energy_Map.shape[0]):
        for m in range(energy_Map.shape[1]):
            choices = energy_cum[n - 1, m:m + 3]
            smallest_m = choices.argmin()

            energy_cum[n, m + 1] = energy_Map[n, m] + choices[smallest_m]
            pointer[n, m] = smallest_m

    # extract n_seam seams
    extracted_seam = []
    column = energy_cum[-1, 1:-1].argmin()
    width = energy_Map.shape[0]
    for w in range(width - 1, 0, -1):
        extracted_seam.append(column)
        column += pointer[w, column] - 1
    extracted_seam.append(column)

    return np.array(extracted_seam[::-1], dtype=INDEX_DTYPE)


def removeVerSeam(image, seam):
    """
        input image is in color, not gray
        return an image with one vertical seam removed
    """

    shape = list(image.shape)
    shape[1] -= 1
    image_seamRemove = np.zeros(shape, dtype=image.dtype)

    for i in range(image.shape[0]):
        image_seamRemove[i, 0:seam[i]] = image[i, 0:seam[i]]
        image_seamRemove[i, seam[i]:] = image[i, seam[i] + 1:]

    return image_seamRemove


def removeHorSeam(image, seam):
    """
        input image is in color, not gray
        return an image with one vertical seam removed
    """

    shape = list(image.shape)
    shape[0] -= 1
    image_seamRemove = np.zeros(shape, dtype=image.dtype)

    for i in range(image.shape[1]):
        image_seamRemove[0:seam[i], i] = image[0:seam[i], i]
        image_seamRemove[seam[i]:, i] = image[seam[i] + 1:, i]

    return image_seamRemove
