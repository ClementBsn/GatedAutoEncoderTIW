#!/usr/bin/python3

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.python import debug as tf_debug
import argparse

import fonc_denoising as de
import util_affich as aff
import util_img as uimg

# EXPERIENCE 1 : 
# - type de réseau utilisé : gae entraîné sur des rotations
# - params du réseau : 400 factor units, 40 mapping units
# - données utilisées : 100 paires d'images (x,y) avec y généré par une 
#   rotation depuis x.
#   Angles = -50,-49,...,49,50 (0 exclu)
# - mesure : comparaison des couches factorielles unies entre chaque paire
# - attendu : les valeurs factorielles mesurées doivent êre conjuguées,
#   càd qu'elles ont les même inner products et des cross products de 
#   valeur opposées pour des rotations d'angles opposés
#   Le contenu de la couche factorielle telle qu'elle est définie dans 
#   le gae_tiw se compose pour la première moitié des valeurs inner et 
#   pour la seconde des valeurs cross. Rappel de la forme de ces produits :
#          - inner product, cas binaire (np.inner) : fx . fy
#            = [fx1,fx2] . [fy1,fy2]
#            = fx1 * fy1 + fx2 * fy2
#          - cross product, cas binaire (np.cross) : fx x fy
#            = [fx1,fx2] x [fy1,fy2]
#            = fx1 * fy2 - fx2 * fy1
parser = argparse.ArgumentParser(description='Lance l\'expérience 2')
parser.add_argument('-f',dest='REP_MOD', metavar='model_folder', type=str, \
                    help="Le chemin relatif du dossier contenant le modèle à tester", \
                    default="../modele_exp1")
parser.add_argument('-m', '--nmoy', dest="N_MOY", metavar="N", type=int, \
                    help="Le nombre de couples d'images sur lequel on moyenne pour chaque rotation", \
                    default=100)
parser.add_argument('-t', '--tailleimg', dest="IMG_SIZE", metavar="image_size", type=int, \
                    help="Taille en pixels des images", default=13)

args = parser.parse_args()

REP_MOD = args.REP_MOD
N_MOY = args.N_MOY
IMG_SIZE = args.IMG_SIZE
NOM_MOD = "modele.meta"
inner_vals, cross_vals = [], []

print("Expérience 1 avec les paramètres suivants :")
print("    Dossier du modèle :",REP_MOD)
print("    Nombre d'exemples pour la moyenne :",N_MOY)
print("    Taille des images :",IMG_SIZE)

for i in range(2):

    with tf.Session() as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # reconstruction du modèle
        print("Reconstruction du modèle...")
        modele = tf.train.import_meta_graph(REP_MOD + "/" + NOM_MOD)
        modele.restore(sess, tf.train.latest_checkpoint(REP_MOD))
        
        # récupération des opérateurs utiles
        cor_x = tf.get_collection("cor_x")[0]
        cor_y = tf.get_collection("cor_y")[0]
        f_cor = tf.get_collection("f_cor")[0]
        print("Modèle prêt. ")
        
        f_cor_vals = []
        
        for ang in range(-50,51):
            if ang==0:
                continue
                #On ignore la rotation nulle
                
            #Génération de 100 paires pour cet angle
            cor_donnees_x = [];
            cor_donnees_y = [];
            for i in range(N_MOY):
                x, y = uimg.get_paire_rot(IMG_SIZE,ang)
                moy_x, std_x = x.mean(), x.std()
                x = (x - moy_x) / std_x
                y = (y - moy_x) / std_x
                cor_donnees_x.append(x.reshape(-1))
                cor_donnees_y.append(y.reshape(-1))
                
            # calcul des valeurs des matrices factorielles de chaque paire
            f_cor_val = sess.run(f_cor,                      \
                                  feed_dict= {                \
                                      cor_x:np.array(cor_donnees_x).T,    \
                                      cor_y:np.array(cor_donnees_y).T})
            
            #On moyenne sur chaque paire
            f_cor_vals.append(f_cor_val.mean(1))
        # print(f_cor_vals[0])
        f_cor_vals = np.array(f_cor_vals)
        print("Dims f : " + str(f_cor_vals.shape))

    # séparation des valeurs inner et cross
    # théoriquement pour L=400, les 200 premières colonnes de f_cor_vals
    # correspondent aux valeurs inner et les 200 suivantes aux cross
    L = f_cor_vals.shape[1]
    inner_vals.append(f_cor_vals[:,:L//2])
    cross_vals.append(f_cor_vals[:,L//2:L])

# coefficients de corrélation (Pearson) stockés sous la forme d'une liste
# de 100 vecteurs contenant 100 coefficients de corrélation, dans l'ordre :
# [[corr_coefs-50vsall],[corr_coefs-49vsall],...,[corr_coefs50vsall]]
# (rappel : 0 exclu)
print("Calcul des coefficients de corrélation...")
corr_inner, corr_cross = [], []
dist_inner, dist_cross = [], []
a=0
# pour chaque paire possible de couche de facteurs
for f1_inner_cour, f1_cross_cour in zip(inner_vals[0],cross_vals[0]):
    # coefficients de corrélation f1 vs all pour inner et cross
    corr_iner_f1, corr_cross_f1 = [], []
    dist_inner_f1, dist_cross_f1 = [], []
    a+=1
    b=0
    for f2_inner_cour, f2_cross_cour in zip(inner_vals[1],cross_vals[1]):
        b+=1
        # on calcule le coefficent de corrélation pour inner et cross
        corr_coef_inner = np.corrcoef([f1_inner_cour, f2_inner_cour])[0][1]
        corr_coef_cross = np.corrcoef([f1_cross_cour, f2_cross_cour])[0][1]
        dist_coef_inner = np.linalg.norm(f1_inner_cour-f2_inner_cour,2)
        dist_coef_cross = np.linalg.norm(f1_cross_cour-f2_cross_cour,2)
        #~ print(np.isfinite(f1_cross_cour).all())
        corr_iner_f1.append(corr_coef_inner)
        corr_cross_f1.append(corr_coef_cross)
        dist_inner_f1.append(dist_coef_inner)
        dist_cross_f1.append(dist_coef_cross)
    
    corr_inner.append(corr_iner_f1)
    corr_cross.append(corr_cross_f1)
    dist_inner.append(dist_inner_f1)
    dist_cross.append(dist_cross_f1)

corr_inner = np.array(np.flipud(corr_inner))
corr_cross = np.array(np.flipud(corr_cross))
dist_inner = np.array(np.flipud(dist_inner))
dist_cross = np.array(np.flipud(dist_cross))

print("Calcul des coefficients de corrélation terminé. ")
print("Dim inner : " + str(corr_inner.shape))
print("Dim cross : " + str(corr_cross.shape))

plt.figure()

plt.subplot(121)
plt.imshow(corr_inner, cmap=cm.bwr_r, vmin=-1, vmax=1, \
           extent=[-50,50,-50,50])
plt.title("Corrélations inner products")
plt.xlabel("Angle de rotation (en degrés)")
plt.ylabel("Angle de rotation (en degrés)")

#plt.show()

plt.subplot(122)
plt.imshow(corr_cross, cmap=cm.bwr_r, vmin=-1, vmax=1, \
           extent=[-50,50,-50,50])
plt.title("Corrélations cross products")
plt.xlabel("Angle de rotation (en degrés)")
plt.ylabel("Angle de rotation (en degrés)")
plt.colorbar(boundaries=np.linspace(-1,1,cm.bwr_r.N,endpoint=True), \
             ticks=[-1, 0, 1])
plt.show()


plt.figure()

plt.subplot(121)
plt.imshow(dist_inner, cmap=cm.plasma, \
           extent=[-50,50,-50,50])
plt.title("Distances inner products")
plt.xlabel("Angle de rotation (en degrés)")
plt.ylabel("Angle de rotation (en degrés)")

#plt.show()

plt.subplot(122)
plt.imshow(dist_cross, cmap=cm.plasma, \
           extent=[-50,50,-50,50])
plt.title("Distances cross products")
plt.xlabel("Angle de rotation (en degrés)")
plt.ylabel("Angle de rotation (en degrés)")
plt.show()
