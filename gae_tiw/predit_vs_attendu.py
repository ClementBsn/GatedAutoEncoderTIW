import numpy as np
import tensorflow as tf
import math
import os

import fonc_denoising as de
import util_affich as aff

REP_STOCK_IMG = "exp1" # répertoire contenant les fichiers d'images
NOM_FICH = "valid.npz" # nom fichier à utiliser
REP_MOD = "resultats/L400_map40_epochs1000_tlot100_corrupt0.3"
NOM_MOD = "modele.meta"
REP_RESULTATS = "retrain"
NB_EX_MONTRER = 10

# PARAMETRES DU GAE
TAILLE_IMG = 13
NB_EPOCHS = 1           # nombre d'entraînements lors d'un apprentissage
MINIBATCH_SIZE = 100    # taille d'un lot d'entraînement
                        # d'apprentissage utilisé lors d'un entraînement
NIV_CORRUPT = 0.3       # le taux de corruption des images x et y

print("Chargement des données...")
donnees = np.load(REP_STOCK_IMG + "/" + NOM_FICH)
donnees_x, donnees_y = donnees["x1"], donnees["y1"]

NB_EXEMPLES = donnees_x.shape[0]
indices_alea = np.arange(NB_EXEMPLES)
np.random.shuffle(indices_alea)
print(indices_alea)

# normalisation des images pour chaque pixel
moy_px_x, moy_px_y = donnees_x.mean(0), donnees_y.mean(0)
std_px_x, std_px_y = donnees_x.std(0), donnees_y.std(0)
donnees_x = (donnees_x - moy_px_x) / std_px_x
donnees_y = (donnees_y - moy_px_y) / std_px_y

# transposée pour obtenir les images en colonne
donnees_x, donnees_y = donnees_x.T[:,indices_alea], donnees_y.T[:,indices_alea]

# corruption des images
print(donnees_x)
print(donnees_x.shape)
cor_donnees_x, cor_donnees_y = donnees_x.copy(), donnees_y.copy()
print(cor_donnees_x.shape)
# print(cor_donnees_x.shape[1])
for i in range(cor_donnees_x.shape[1]):
    de.zero_mask(cor_donnees_x[:,i], NIV_CORRUPT)
    de.zero_mask(cor_donnees_y[:,i], NIV_CORRUPT)
print("Données chargées. ")


with tf.Session() as sess:
    # reconstruction du modèle
    print("Reconstruction du modèle...")
    modele = tf.train.import_meta_graph(REP_MOD + "/" + NOM_MOD)
    modele.restore(sess, tf.train.latest_checkpoint(REP_MOD))
    
    # récupération des opérateurs utiles
    y = tf.get_collection("y")[0]
    cor_x = tf.get_collection("cor_x")[0]
    cor_y = tf.get_collection("cor_y")[0]
    rec = tf.get_collection("rec")[0]
    fonc_cout = tf.get_collection("fonc_cout")[0]
    optim = tf.get_collection("optim")[0]
    saver = tf.train.Saver()
    print("Modèle prêt. ")
    
    # pour affichage de NB_EX_MONTRER images reconstruites
    img_predites = sess.run(                                       \
            rec,                                             \
            feed_dict={                                           \
                cor_x: cor_donnees_x[:,:NB_EX_MONTRER], \
                cor_y: cor_donnees_y[:,:NB_EX_MONTRER]})


# formatage pour affichage
# tranposition
donnees_y    = donnees_y.T
img_predites = img_predites.T
# "dénormalisation"
donnees_y    = (donnees_y * std_px_y) + moy_px_y
img_predites = (img_predites * std_px_y) + moy_px_y

img_attendues = aff.multiVecToMultiMat(             \
                    donnees_y[:NB_EX_MONTRER],     \
                    TAILLE_IMG, TAILLE_IMG)
img_predites = aff.multiVecToMultiMat(         \
                    img_predites,         \
                    TAILLE_IMG, TAILLE_IMG)

aff.sauvegarderGrilleImages2L(                     \
    NB_EX_MONTRER, img_attendues, img_predites, \
    "attendu_vs_predit", "png")
print("Des exemples de reconstruction ont été sauvegardés. ")
