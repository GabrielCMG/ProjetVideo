import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve as filter2
import cv2


def HS(im1, im2, alpha, Niter):
    """
    Fonction implémentant la méthode de flot optique proposée par B.K.P. Horn et B.G. Schunck.

    :param im1: image à t
    :param im2: image à t+dt
    :param alpha: constante de régularisation
    :param Niter: nombre d'itérations
    :return: flot optique
    """

    # Initialisation de la vitesse
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    # Initialisation des vecteurs de flot optique
    U = uInitial
    V = vInitial

    # Estimation des dérivées
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Noyau de moyennage
    kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                       [1 / 6, 0, 1 / 6],
                       [1 / 12, 1 / 6, 1 / 12]], float)

    # Itérations pour réduire l'erreur
    for _ in range(Niter):
        # Calcul de la moyenne locale des vecteurs de flot
        uAvg = filter2(U, kernel)
        vAvg = filter2(V, kernel)
        der = (fx * uAvg + fy * vAvg + ft) / (4*alpha ** 2 + fx ** 2 + fy ** 2)
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


def computeDerivatives(im1, im2):
    """
    Fonction calculant les dérivées de l'image (en x, en y et en t)

    :param im1: image à t
    :param im2: image à t+dt
    :return: dérivées de l'image
    """
    # Construction des noyaux pour calculer les dérivées
    kernelX = np.array([[-1, 1], [-1, 1]]) * .25  # Noyau pour calculer d/dx
    kernelY = np.array([[-1, -1], [1, 1]]) * .25  # Noyau pour calculer d/dy
    kernelT = np.ones((2, 2)) * .25

    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)

    # ft = im2 - im1
    ft = filter2(im1, -kernelT) + filter2(im2, kernelT)

    return fx, fy, ft


cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(cv2.samples.findFile("Poubelle/vtest3.mp4"))
_, old_frame = cap.read()

while 1:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcul du flot optique
    U, V = HS(old_frame_gray, frame_gray, 15, 20)

    M = np.sqrt(U ** 2 + V ** 2)
    M = 10*np.log10(M)

    # th = np.quantile(M, 0.99)
    # M[M <= th] = 0

    plt.clf()
    plt.imshow(M)
    plt.colorbar()
    plt.pause(0.001)

    k = 0xff
    if k == 27:
        break

    # Mise à jour de l'image de l'itération précédente
    old_frame = frame.copy()

cv2.destroyAllWindows()
