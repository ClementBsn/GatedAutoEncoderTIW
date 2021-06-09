#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf

import fonc_denoising as de
import util_affich as aff
from dae import *

# on utilise ici les données MNIST disponibles sur le site de Yann LeCun :
# il s'agit d'un ensemble de 70000 images 28px * 28ps décomposé en 3 
# sous-ensembles : 
# - mnist.train : 55000 données
# - mnist.test : 10000 données
# - mnist.validation : 5000 données
# Chacun de ces ensembles contient une matrice image et une matrice labels
# correspondant respectivement aux images fournies en entrée et aux étiquettes
# attribuées à celles-ci. A noter que la représentation d'une étiquette qui 
# normalement est un valeur allant de 0 à 9 est ici présente sous la forme 
# d'un vecteur V où V[i] = 0 si l'image est censée appartenir à la classe ici
# et 0 sinon. 
print("Recuperation des donnees MNIST...")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

print("Initialisation du modele...")
# réglages :
TAUX_APP    = 0.1  # le taux d'apprentissage
NB_ENTRAIN  = 1000 # le nombre de fois que l'on entraîne le modèle
TAILLE_LOT  = 250  # nombre d'instances dans un lot
DECAY       = 0.4
NB_EXEMPLES = 20   # nombre d'exemples à montrer

# caractéristiques d'une image
LONG_IMG    = 28
HAUT_IMG    = 28
NB_PIXELS   = LONG_IMG * HAUT_IMG

# tailles des couches de l'autoencoder :
S1, S2, S3 = NB_PIXELS, int(NB_PIXELS/3), int(NB_PIXELS/6)
S4, S5 = S2, S1

# nombre de lots total dans MNIST
# mnist.train.num_examples désigne le nombre d'exemples (55000 en théorie)
NB_LOTS = int(mnist.train.num_examples / TAILLE_LOT)

# définition de la représentation de l'entrée
X = tf.placeholder("float", [None, S1])
XCorrompu = tf.placeholder("float", [None, S1])

# définition des matrices de poids de l'encodeur et du décodeur, 
# initialisées aléatoirement selon la loi normale
encodW1 = tf.Variable(tf.random_normal([S1,S2]))
encodW2 = tf.Variable(tf.random_normal([S2,S3]))
decodW1 = tf.Variable(tf.random_normal([S3,S4]))
decodW2 = tf.Variable(tf.random_normal([S4,S5]))
encodW = [encodW1, encodW2]
decodW = [decodW1, decodW2]

# idem pour les biais
encodB1 = tf.Variable(tf.random_normal([S2]))
encodB2 = tf.Variable(tf.random_normal([S3]))
decodB1 = tf.Variable(tf.random_normal([S4]))
decodB2 = tf.Variable(tf.random_normal([S5]))
encodB = [encodB1, encodB2]
decodB = [decodB1, decodB2]

# construction du modèle
encoderOp = coder(XCorrompu, encodW, encodB)
decoderOp = coder(encoderOp, decodW, decodB)

# finalement :
# decoderOp = sortie prédite à comparer avec l'entrée X
yPred = decoderOp

# on utilise une fonction de coût de type erreur quadratique moyenne
foncCout = tf.reduce_mean(tf.pow(decoderOp - X, 2))

# valeurs obtenues pour la fonction de coût pour affichage
foncCoutVal = []

# on choisit d'utiliser une descente de gradient classique par lot pour  
# minimiser la fonction de coût
optimiseur = tf.train.RMSPropOptimizer(TAUX_APP,decay=DECAY).minimize(foncCout)
# optimiseur = tf.train.GradientDescentOptimizer(TAUX_APP).minimize(foncCout)
#GradientDescentOptimizer
#RMSPropOptimizer

# initialisation des variables
init = tf.global_variables_initializer()

#####################################################################
result_stats = []
for i_app in range(4):
    result_stats.append([])
    for i_decay in range(1,10):
        optimiseur = tf.train.RMSPropOptimizer(TAUX_APP/(10.0**float(i_app)),decay=float(i_decay)/10.0).minimize(foncCout)
        # optimiseur = tf.train.GradientDescentOptimizer(TAUX_APP).minimize(foncCout)
        #GradientDescentOptimizer
        #RMSPropOptimizer

        # initialisation des variables
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            print("Entrainement du modele..."+str(TAUX_APP/(10.0**float(i_app)))+" "+str(float(i_decay)/10.0))
            sess.run(init) # exécution de la phase d'initialisation des variables
            
            # entraînement NB_ENTRAIN fois
            for i in range(NB_ENTRAIN):
                # pour chaque lot de MNIST
                # for j in range(NB_LOTS):
                # on récupère aléatoirement un lot
                lotX, _ = mnist.train.next_batch(TAILLE_LOT)
                lotXCorrompu = lotX
                de.zero_mask(lotXCorrompu, 0.3)
                
                # puis on lance la descente de gradient sur le lot
                # (on récupère également le coût)
                _, cout = sess.run([optimiseur, foncCout], feed_dict={X: lotX, XCorrompu: lotXCorrompu})
                if (i+1)%100==0:
                    print("Taux d'apprentissage : "+str(TAUX_APP/(10.0**float(i_app))))
                    print("Decay                : "+str(float(i_decay)/10.0))
                    print("Entrainement",(i+1),"/",NB_ENTRAIN)
                    print("Cout du dernier lot :"+str(cout))
        result_stats[i_app].append(cout)
        print("result_stats="+str(result_stats))

            # print("Tests du modele sur", NB_EXEMPLES,"exemples...")
            # # on applique encodage et décodage sur NB_EXEMPLES images de l'ensemble
            # # de test
            # encoderEtDecoder = sess.run(yPred, feed_dict={X: mnist.test.images[:NB_EXEMPLES]})
# 
# 
# print("Affichage des tests.")
# # on récupère NB_EXEMPLES images (entrée et sortie)
# imgEntree = aff.multiVecToMultiMat(mnist.test.images[:NB_EXEMPLES], \
                                   # HAUT_IMG, LONG_IMG)
# imgSortie = aff.multiVecToMultiMat(encoderEtDecoder[:NB_EXEMPLES], \
                                   # HAUT_IMG, LONG_IMG)
# 
# # que l'on affiche dans une grille de 2 lignes : 
# # 1 pour les images en entrée et 1 pour les images en sortie
# aff.afficherGrilleImages2L(NB_EXEMPLES, imgEntree, imgSortie)
# 
# aff.afficherFoncCout(foncCoutVal)

input("Appuyez sur n'importe quelle touche pour terminer...")

