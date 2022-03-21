import cv2
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
from numpy.core.fromnumeric import size
import colorsys

from numpy.lib.function_base import angle





def main():
################################### user input ###################################
    # error check user input
    if len(sys.argv) != 2:
        print('usage: python3 a5.py <input_image_name>\n')
        return

    # read in the image in RGB
    img = cv2.imread(sys.argv[1], 0)

    # displaying the original image
    cv2.imshow('Image', img)
    cv2.waitKey(0)



################################### initializing ###################################
    # initializing sobel kernels
    Gx = np.matrix('-1 0 1; -2 0 2; -1 0 1')
    Gy = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')

    # using the scaled kernels
    Gx = Gx/4
    Gy = Gy/4


    # pads the image
    img = np.pad(img, 1, "constant")

    # creates images the same size as the padded image with all zeros
    newImgx = np.zeros_like(img)
    newImgy = np.zeros_like(img)
    magImg = np.zeros_like(img)
    angleImg = np.zeros_like(img) 
    newImgDoublex = np.zeros_like(img)
    newImgDoubley = np.zeros_like(img)
    magImgDouble = np.zeros_like(img)
    gray2RGB = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(gray2RGB, cv2.COLOR_RGB2HSV)



################################### loop ###################################
    # go through every pixel in the image, excluding padding
    for row in range(1, img.shape[0]-1):
        for col in range(1, img.shape[1]-1):
            # get the 3x3 patch
            patch = img[row-1:row+2,col-1:col+2]

            ################################### part a ###################################
            # calculate
            ix = np.multiply(patch, Gx)
            ixSum = np.sum(ix)
            iy = np.multiply(patch, Gy)
            iySum = np.sum(iy)

            # initialize the new image
            newImgx[row][col] = ixSum
            newImgy[row][col] = iySum


            ################################### part b ###################################
            # compute the magnitude in range of 0-255*sqrt(2)
            mag = np.sqrt(np.square(ixSum) + np.square(iySum))

            # initialize new image
            magImg[row][col] = mag


            ################################### part c ###################################
            # error checking for angle tangent
            if ixSum == 0 and iySum == 0:
                theta = 0

            # computing the angles in degrees 0-360
            theta = math.degrees(np.arctan2(ixSum, iySum))

            # if the angle is negative
            if(theta < 0): 
                theta = theta + 360

            # initialize new image
            angleImg[row][col] = theta

            hsv[row][col] = angleImg[row][col], 255, magImg[row][col]


            ################################### part ii ###################################
            # calculate
            # ix & iy are already multiplied once so multiply again
            ixDouble = np.multiply(Gx, ix)
            ixDoubleSum = np.sum(ixDouble)
            iyDouble = np.multiply(Gy, iy)
            iyDoubleSum = np.sum(iyDouble)

            # initialize the new image
            newImgDoublex[row][col] = ixDoubleSum
            newImgDoubley[row][col] = iyDoubleSum

            # compute the magnitude in range of 0-255*sqrt(2)
            magDouble = np.sqrt(np.square(ixDoubleSum) + np.square(iyDoubleSum))

            # initialize new image
            magImgDouble[row][col] = magDouble



    ################################### part a ###################################
    # get correct range
    ex = newImgx + 255 / (510 / 255)
    ey = newImgy + 255 / (510 / 255)

    # make sure the image is unsigned integers
    ex = ex.astype(np.uint8)
    ey = ey.astype(np.uint8)

    # displaying the ix image
    cv2.imshow('Ix', ex)
    cv2.waitKey(0)

    # displaying the iy image
    cv2.imshow('Iy', ey)
    cv2.waitKey(0)



    ################################### part b ###################################
    # get correct range for magnitue
    eMag = magImg / ((255 * np.sqrt(2)) / 255)

    # make sure the image is unsigned integers
    eMag = eMag.astype(np.uint8)

    # displaying the mag image
    cv2.imshow('Magnitude Image', eMag)
    cv2.waitKey(0)



    ################################### part c ###################################
    # get correct range for angle
    # eAng = angleImg / (360 / 255)

    # # make sure the image is unsigned integers
    # eAng = eAng.astype(np.uint8)

    # gray2RGB = cv2.cvtColor(eAng,cv2.COLOR_GRAY2RGB)
    # hsv = cv2.cvtColor(gray2RGB, cv2.COLOR_RGB2HSV)

    # for row in range(1, img.shape[0]-1):
    #     for col in range(1, img.shape[1]-1):
    #         hsv[row][col] = angleImg[row][col], 255, magImg[row][col]

    hsv = hsv.astype(np.uint8)

    # displaying the mag image
    cv2.imshow('Angle Image', hsv)
    cv2.waitKey(0)



    ################################### part ii ###################################
    # get correct range
    exDouble = newImgDoublex + 255 / (510 / 255)
    eyDouble = newImgDoubley + 255 / (510 / 255)

    # make sure the image is unsigned integers
    exDouble = exDouble.astype(np.uint8)
    eyDouble = eyDouble.astype(np.uint8)

    # displaying the ix image
    cv2.imshow('Ix Double', exDouble)
    cv2.waitKey(0)

    # displaying the iy image
    cv2.imshow('IyDouble', eyDouble)
    cv2.waitKey(0)

    # get correct range for magnitue
    eMagDouble = magImgDouble / ((255 * np.sqrt(2)) / 255)

    # make sure the image is unsigned integers
    eMagDouble = eMagDouble.astype(np.uint8)

    # displaying the mag image
    cv2.imshow('Magnitude Image Double', eMagDouble)
    cv2.waitKey(0)





main()