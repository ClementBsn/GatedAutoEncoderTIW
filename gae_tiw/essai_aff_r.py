import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import fonc_denoising as de
import util_affich as aff

TAILLE_IMG = 13
NB_EX_MONTRER = 10 # le nombre d'images prédites à montrer pour comparaison
REP_STOCK_IMG = "exp1" # répertoire contenant les fichiers d'images
NOM_FICH = "valid.npz" # nom fichier à utiliser
REP_MOD = "resultats/sigm/L400_map40_epochs500_tlot100_corrupt0.3"
NOM_MOD = "modele.meta"

donnees = np.load(REP_STOCK_IMG + "/" + NOM_FICH)

donnees_x, donnees_y = donnees["x"].T, donnees["y"].T

# corruption des images
cor_donnees_x, cor_donnees_y = donnees_x.copy(), donnees_y.copy()
print(cor_donnees_x.shape[1])
# for i in range(cor_donnees_x.shape[1]):
    # de.zero_mask(cor_donnees_x[:,i], 0.3)
    # de.zero_mask(cor_donnees_y[:,i], 0.3)

with tf.Session() as sess:
    # reconstruction du modèle
    modele = tf.train.import_meta_graph(REP_MOD + "/" + NOM_MOD)
    modele.restore(sess, tf.train.latest_checkpoint(REP_MOD))
    
    cor_x = tf.get_collection("cor_x")[0]
    cor_y = tf.get_collection("cor_y")[0]
    rec = tf.get_collection("rec")[0]
    
    for i in range(NB_EX_MONTRER):
        # i:i+1 permet d'obtenir une matrice à la place d'un vecteur
        img_attendue = donnees_y[:,i:i+1].reshape(TAILLE_IMG,TAILLE_IMG)
        img_predite = (sess.run(                                    \
                        rec, feed_dict= {   \
                            cor_x:cor_donnees_x[:,i:i+1],   \
                            cor_y:cor_donnees_y[:,i:i+1]})) \
                       .reshape(TAILLE_IMG,TAILLE_IMG)
        print("Img " + str(i))
        print(img_attendue)
        print(img_predite)
        
        plt.figure()
        fig, grille = plt.subplots(2,1)
        grille[0].imshow(img_attendue, cmap='plasma')
        grille[0].axis("off")
        grille[1].imshow(img_predite, cmap='plasma')
        grille[1].axis("off")
        plt.savefig("attendu_vs_predit" + str(i))

# img_attendues = aff.multiVecToMultiMat(             \
                    # donnees_y.T[:NB_EX_MONTRER],     \
                    # TAILLE_IMG, TAILLE_IMG)
# img_predites = aff.multiVecToMultiMat(         \
                    # img_predites.T,         \
                    # TAILLE_IMG, TAILLE_IMG)

# aff.sauvegarderGrilleImages2L(                     \
    # NB_EX_MONTRER, img_attendues, img_predites, \
    # "attendu_vs_predit", "png")