#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

import argparse
import sys

import fonc_denoising as de
import util_affich as aff
import util_img as uimg

# EXPERIENCE 1 : 
# - type de réseau utilisé : gae entraîné sur des rotations
# - params du réseau : 400 factor units, 40 ou 5 mapping units, entrainé
#   sur des angles de valeurs multiples de 10 (en degrés)
# - données utilisées : 100 paires d'images (x,y) avec y généré par une 
#   rotation depuis x.
#   Angles = -50,-49,...,49,50 (0 exclu)
# - mesure : comparaison des couches factorielles unies entre chaque paire
# - attendu : Le réseau utilise une unité de mapping pour chaque rotation 
#   du jeu d'apprentissage lorsque c'est possible, et adapte l'encodage
#   des données à un nombre d'unités de mapping plus faible lorsque cela 
#   est nécessaire
parser = argparse.ArgumentParser(description='Lance l\'expérience 2')
parser.add_argument('-f',dest='REP_MOD', metavar='model_folder', type=str, \
                    help="Le chemin relatif du dossier contenant le modèle à tester", \
                    default="../modele_exp2_40mappings")
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
map_vals = []

for i in range(2):

    with tf.Session() as sess:
        # reconstruction du modèle
        print("Reconstruction du modèle...")
        sys.stdout.flush()
        modele = tf.train.import_meta_graph(REP_MOD + "/" + NOM_MOD)
        modele.restore(sess, tf.train.latest_checkpoint(REP_MOD))
        
        # récupération des opérateurs utiles
        cor_x = tf.get_collection("cor_x")[0]
        cor_y = tf.get_collection("cor_y")[0]
        f_cor = tf.get_collection("f_cor")[0]
        m = tf.get_collection("m")[0]
        print("Modèle prêt. ")
        sys.stdout.flush()
        
        map_layers = []
        
        for ang in range(-50,51):
            #~ if ang==0:
                #~ continue
                #~ #On ignore la rotation nulle
                
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
            m_val = sess.run(m,                      \
                        feed_dict= {                \
                        cor_x:np.array(cor_donnees_x).T,    \
                        cor_y:np.array(cor_donnees_y).T})
            
            #On moyenne sur chaque paire
            map_layers.append(m_val.mean(1))
        # print(map_layers[0])
        map_layers = np.array(map_layers)
        print("Dims map_layers : " + str(map_layers.shape))
        sys.stdout.flush()

    # séparation des valeurs inner et cross
    # théoriquement pour L=400, les 200 premières colonnes de map_layers
    # correspondent aux valeurs inner et les 200 suivantes aux cross
    map_vals.append(map_layers)

# coefficients de corrélation (Pearson) stockés sous la forme d'une liste
# de 100 vecteurs contenant 100 coefficients de corrélation, dans l'ordre :
# [[corr_coefs-50vsall],[corr_coefs-49vsall],...,[corr_coefs50vsall]]
# (rappel : 0 exclu)
print("Courbes d'activation de la couche de mappings en fonction de l'angle de rotation")
sys.stdout.flush()
toPlot = np.array(map_vals[0]).T
plt.figure()
for neuron in toPlot:
    plt.plot(range(-50,51),neuron)
plt.title("Courbes d'activation moyenne de la couche de mappings \n en fonction de l'angle de rotation")
plt.xlabel("Angle de rotation")
plt.xlim([-50,50])
plt.ylabel("Activation")


print("Calcul des distances entre les activations moyennes")
sys.stdout.flush()
distances = []


# pour chaque paire possible de couche de facteurs
for map1_layer in map_vals[0]:
    # coefficients de corrélation f1 vs all pour inner et cross
    distance1 = []
    
    for map2_layer in map_vals[1]:
        # on calcule le coefficent de corrélation pour inner et cross
        distance = np.linalg.norm(map1_layer-map2_layer, 2)
        distance1.append(distance)
    
    distances.append(distance1)

distances = np.array(np.flipud(distances))

print("Calcul des distances terminé. ")
print("Dim inner : " + str(distances.shape))
sys.stdout.flush()

dist_max = np.amax(distances)

plt.figure()
plt.imshow(distances, cmap=cm.plasma, \
           extent=[-50,50,-50,50])
plt.title("Distances (en norme quadratique) entre les activations \n moyennes de la couche de mappings")
plt.xlabel("Angle de rotation (en degrés)")
plt.ylabel("Angle de rotation (en degrés)")
plt.colorbar(boundaries=np.linspace(0,dist_max,cm.bwr_r.N,endpoint=True), \
             ticks=[0, dist_max/2, dist_max])
plt.show()
