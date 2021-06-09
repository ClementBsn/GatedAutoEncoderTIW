import numpy as np
import tensorflow as tf
import math
import os

import fonc_denoising as de
import util_affich as aff

# EXPERIENCE 1 : 
# - type de réseau utilisé : gae entraîné sur des rotations
# - params du réseau : 400 factor units, 40 mapping units
# - données utilisées : 100 paires d'images (x,y) avec y généré par une 
#   rotation depuis x.
#   Angles = -50,-49,...,49,50 (0 exclu)
# - mesure : comparaison des couches factorielles unies entre chaque paire
# - attendu : les valeurs factorielles mesurées doivent êre conjuguées,
#   càd qu'elles ont les même inner products et des cross products de 
#   valeur opposées
#   Le contenu de la couche factorielle telle qu'elle est définie dans 
#   le gae_tiw se compose pour la première moitié des valeurs inner et 
#   pour la seconde des valeurs cross. Rappel de la forme de ces produits :
#          - inner product, cas binaire (np.inner) : fx . fy
#            = [fx1,fx2] . [fy1,fy2]
#            = fx1 * fy1 + fx2 * fy2
#          - cross product, cas binaire (np.cross) : fx x fy
#            = [fx1,fx2] x [fy1,fy2]
#            = fx1 * fy2 - fx2 * fy1
REP_STOCK_IMG = "exp1" # répertoire contenant les fichiers d'images
NOM_FICH = "train.npz" # nom fichier à utiliser
REP_MOD = "resultats/L400_map40_epochs100_tlot100_corrupt0.3"
NOM_MOD = "modele.meta"
REP_RESULTATS = "retrain"
MOD_RESULTATS = REP_RESULTATS + "/L400_map40_epochs100_tlot100_corrupt0.3"
NB_EX_MONTRER = 10

# PARAMETRES DU GAE
TAILLE_IMG = 13
NB_EPOCHS = 100        # nombre d'entraînements lors d'un apprentissage
MINIBATCH_SIZE = 100    # taille d'un lot d'entraînement
                        # d'apprentissage utilisé lors d'un entraînement
NIV_CORRUPT = 0.3       # le taux de corruption des images x et y

# chargement des données (paires rangées dans l'ordre croissant des
# angles, de -50 à 50, 0 exclu)
print("Chargement des données...")
donnees = np.load(REP_STOCK_IMG + "/" + NOM_FICH)
donnees_x, donnees_y = donnees["x"], donnees["y"]

# normalisation des images pour chaque pixel
donnees_x -= donnees_x.mean(0)
donnees_y -= donnees_y.mean(0)
donnees_x /= donnees_x.std(0)
donnees_y /= donnees_y.std(0)

# transposée pour obtenir les images en colonne
donnees_x, donnees_y = donnees_x.T, donnees_y.T

# corruption des images
cor_donnees_x, cor_donnees_y = donnees_x.copy(), donnees_y.copy()
# print(cor_donnees_x.shape[1])
for i in range(cor_donnees_x.shape[1]):
    de.zero_mask(cor_donnees_x[:,i], NIV_CORRUPT)
    de.zero_mask(cor_donnees_y[:,i], NIV_CORRUPT)
print("Données chargées. ")

fonc_cout_vals = []

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
    
    print("Entrainement du GAE...")
    NB_EXEMPLES = donnees_x.shape[1]
        
    # entraînement EPOCHS fois
    for i in range(NB_EPOCHS):
        print("Entrainement",(i+1),"/",NB_EPOCHS)
        
        # indices des exemples rangés aléatoirement
        indices_alea = np.arange(NB_EXEMPLES)
        np.random.shuffle(indices_alea)
        
        # sélection des MINIBATCH_SIZE 1ers exemples
        i = 0
        j = MINIBATCH_SIZE
        
        # tant qu'il reste des exemples pour l'entraînement courant
        coutMoy = 0
        while j < NB_EXEMPLES:
            # on récupère les lots aléatoires courants
            lot_cor_x = cor_donnees_x[:,indices_alea[i:j]]
            lot_cor_y = cor_donnees_y[:,indices_alea[i:j]]
            lot_y = donnees_y[:,indices_alea[i:j]]
            i += MINIBATCH_SIZE
            j += MINIBATCH_SIZE
            # puis on lance la descente de gradient sur le lot
            # (on récupère également le coût)
            with tf.device("/cpu:0"): # optim pour calcul
                _, cout = \
                    sess.run([optim, fonc_cout], \
                        feed_dict={cor_x: lot_cor_x, \
                                   cor_y: lot_cor_y, \
                                   y:     lot_y})
                coutMoy += cout
        coutMoy /= (NB_EXEMPLES / MINIBATCH_SIZE)
            
        print("Cout moyen du dernier entrainement :",coutMoy)
        fonc_cout_vals.append(coutMoy)
    
    print("Entrainement terminé.")
    # on créé un répertoire "resultats" si celui-ci n'existe pas
    try:
        os.mkdir(REP_RESULTATS)
    except OSError:
        pass
    
    # création sur le disque du répertoire stockant le modèle créé
    try:
        os.mkdir(MOD_RESULTATS)
    except OSError:
        pass
        
    # sauvegarde du modèle
    saver.save(sess, MOD_RESULTATS + "/modele")
    print("Le modèle créé a été sauvegardé. ")
    
    # pour affichage de NB_EX_MONTRER images reconstruites
    img_predites = sess.run(                                       \
            rec,                                             \
            feed_dict={                                           \
                cor_x: cor_donnees_x[:,:NB_EX_MONTRER], \
                cor_y: cor_donnees_y[:,:NB_EX_MONTRER]})

aff.sauvegarderFoncCout(fonc_cout_vals,             \
                        MOD_RESULTATS + "/cout",     \
                        "png")
print("Un historique de la fonction de coût a été sauvegardé. ")

# formatage pour affichage
img_attendues = aff.multiVecToMultiMat(             \
                    donnees_y.T[:NB_EX_MONTRER],     \
                    TAILLE_IMG, TAILLE_IMG)
img_predites = aff.multiVecToMultiMat(         \
                    img_predites.T,         \
                    TAILLE_IMG, TAILLE_IMG)

aff.sauvegarderGrilleImages2L(                     \
    NB_EX_MONTRER, img_attendues, img_predites, \
    MOD_RESULTATS + "/attendu_vs_predit", "png")
print("Des exemples de reconstruction ont été sauvegardés. ")
