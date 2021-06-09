import tensorflow as tf
import numpy as np
import scipy.linalg
import math
import os

import fonc_denoising as de
import util_affich as aff

REP_STOCK_IMG = "img" # répertoire contenant les fichiers d'images
NOM_FICH = "rotate_50_13_100000.npz" # nom fichier à utiliser
NB_EX_MONTRER = 10 # le nombre d'images prédites à montrer pour comparaison
donnees = np.load(REP_STOCK_IMG + "/" + NOM_FICH)
donnees_x, donnees_y = donnees["x"].T, donnees["y"].T
TAILLE_IMG= 13

print(donnees_x)
print(donnees_y)

img_x = aff.multiVecToMultiMat(donnees_x.T[:NB_EX_MONTRER], \
                                       TAILLE_IMG, TAILLE_IMG)
img_y = aff.multiVecToMultiMat(donnees_y.T[:NB_EX_MONTRER], \
                                       TAILLE_IMG, TAILLE_IMG)

aff.afficherGrilleImages2L(NB_EX_MONTRER, img_x, img_y)

input("Appuyez sur n'importe quelle touche pour terminer...")

