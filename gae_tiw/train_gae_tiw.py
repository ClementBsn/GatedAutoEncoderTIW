#!/usr/bin/python3

import numpy as np
import math
import argparse

from gae_tiw import GatedAutoencoderTIW

parser = argparse.ArgumentParser(description='Lance l\'expérience 2')
parser.add_argument('-f',dest='REP_MOD', metavar='model_folder', type=str, \
                    help="Le chemin relatif du dossier dans lequel on sauvegarde le modèle", \
                    default="")
parser.add_argument('-t', '--tailleimg', dest="TAILLE_IMG", metavar="image_size", type=int, \
                    help="Taille en pixels des images", default=13)
parser.add_argument('-L', '--factors', dest="L", metavar="n_facteurs", type=int, \
                    help="Taille de la couche de facteurs", default=400)
parser.add_argument('-M', '--mappings', dest="N_MAPPINGS", metavar="n_mapping", type=int, \
                    help="Taille de la couche de mappings", default=40)
parser.add_argument('-E', '--epochs', dest="NB_EPOCHS", metavar="n_epochs", type=int, \
                    help="Nombre d'epochs pour l'entraînement", default=200)
parser.add_argument('-m', '--minibatch', dest="MINIBATCH_SIZE", metavar="minibatch_size", type=int, \
                    help="Taille des minibatchs", default=100)
parser.add_argument('-A', '--apprentissage', dest="TAUX_APP", metavar="taux_apprentissage", type=float, \
                    help="Taux d'apprentissage initial", default=0.005)
parser.add_argument('-C', '--corruption', dest="NIV_CORRUPT", metavar="niveau_corruption", \
                    type=float, help="Taux de corruption", default=0.3)
parser.add_argument('-G', '--gauss', dest="SIGM_FILTRE_GAUSS", metavar="sigma", type=float, \
                    help="Valeur sigma du filtre gaussien appliqué lors du calcul de l'erreur de reconstruction", \
                    default=1)
parser.add_argument('-d', '--dataset', dest="REP_STOCK_IMG", metavar="répertoire", type=str, \
                    help="Dossier contenant le 'train.npz' à utiliser", default="exp1")


args = parser.parse_args()

# PARAMETRES DU GAE
TYPE_X, TYPE_Y = "float64", "float64" # les types des données
TAILLE_IMG = args.TAILLE_IMG
NB_PX = TAILLE_IMG * TAILLE_IMG # le nombre de pixels d'une image
L = args.L
N_MAPPINGS = args.N_MAPPINGS
NB_EPOCHS = args.NB_EPOCHS
MINIBATCH_SIZE = args.MINIBATCH_SIZE
TAUX_APP = args.TAUX_APP
                        # d'apprentissage utilisé lors d'un entraînement
NIV_CORRUPT = args.NIV_CORRUPT
SIGM_FILTRE_GAUSS = args.SIGM_FILTRE_GAUSS
                        # aux images lors du calcul de l'erreur de 
                        # reconstruction
if args.REP_MOD=="":
    REP_MOD = args.REP_STOCK_IMG+"_modele"
else:
    REP_MOD = args.REP_MOD
    

# création du gae
g1 = GatedAutoencoderTIW(TYPE_X, TYPE_Y, TAILLE_IMG, L, N_MAPPINGS, \
                         TAUX_APP, NIV_CORRUPT, NB_EPOCHS, \
                         MINIBATCH_SIZE, SIGM_FILTRE_GAUSS, args.REP_MOD)

# chargement des données images pour apprentissage
REP_STOCK_IMG = args.REP_STOCK_IMG
NOM_FICH = "train.npz" # nom fichier à utiliser
NB_EX_MONTRER = 10 # le nombre d'images prédites à montrer pour comparaison
donnees = np.load(REP_STOCK_IMG + "/" + NOM_FICH)

g1.apprendre(donnees)

input("Appuyez sur n'importe quelle touche pour terminer...")

