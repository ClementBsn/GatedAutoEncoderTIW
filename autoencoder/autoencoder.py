# -*- coding: utf-8 -*-

import tensorflow as tf

def coder(x, W, B):
    """
    Agit comme un encoder ou un decoder selon le W fourni.
    Calcule les paramètres de la couche cachée centrale OU de la couche de
    sortie par feedforward.

    Ex si encoder :
    a0 = X
    a1 = sigmoïde(a0 * encodW1 + encodB1)
    ...
    Jusqu'à la couche centrale cachée représentant les données compressées.

    Ex si decoder :
    a0 = couche centrale cachée
    a1 = sigmoïde(a0 * decodW1 + decodB1)
    ...
    Jusqu'à la couche de sortie.
    :param x: la couche d'entrée de l'autoencoder
    :param W: la liste des matrices des poids
    :param B: la liste des matrices des biais
    """
    aCour = x # unité d'activation de la couche 0 = x
    # pour chaque matrice de poids et de biais
    for w, b in zip(W, B):
        # on calcule l'unité d'activation de la couche courante
        aCour = tf.nn.sigmoid(tf.matmul(aCour, w)+ b)

    return aCour
