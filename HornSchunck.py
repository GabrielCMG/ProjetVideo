#!/usr/bin/env python
"""
examples:

./HornSchunck.py data/box/box
./HornSchunck.py data/office/office
./HornSchunck.py data/rubic/rubic
./HornSchunck.py data/sphere/sphere

"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as filter2
import cv2

FILTER = 7
QUIVER = 5


def HS(im1, im2, alpha, Niter):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Averaging kernel
    kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                       [1 / 6, 0, 1 / 6],
                       [1 / 12, 1 / 6, 1 / 12]], float)

    # Iteration to reduce error
    for _ in range(Niter):
        # %% Compute local averages of the flow vectors
        uAvg = filter2(U, kernel)
        vAvg = filter2(V, kernel)
        # %% common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        # %% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


def computeDerivatives(im1, im2):
    # %% build kernels for calculating derivatives
    kernelX = np.array([[-1, 1],
                        [-1, 1]]) * .25  # kernel for computing d/dx
    kernelY = np.array([[-1, -1],
                        [1, 1]]) * .25  # kernel for computing d/dy
    kernelT = np.ones((2, 2)) * .25

    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)

    # ft = im2 - im1
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft


cap = cv2.VideoCapture(0)

ret, old_frame = cap.read()

while 1:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    U, V = HS(old_frame_gray, frame_gray, 0.1, 10)

    M = np.sqrt(U ** 2 + V ** 2)

    plt.imshow(10*np.log10(M))

    plt.pause(0.001)

    k = 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_frame = frame.copy()

cv2.destroyAllWindows()
