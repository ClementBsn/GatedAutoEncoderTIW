#!/usr/bin/python3

import numpy as np
import os
import argparse

import util_img as uimg

parser = argparse.ArgumentParser(description='Génère un jeu de données pour l\'entraînement')
parser.add_argument('-N', '--n-paires', dest="NB_PAIRES_TRAIN", metavar="nb_paires", \
                    type=int, help="Nombres de paires d'images à générer pour l'entraînement", \
                    default=100000)
parser.add_argument('-t', '--tailleimg', dest="TAILLE_IMG", metavar="image_size", type=int, \
                    help="Taille en pixels des images", default=13)
parser.add_argument('-M', '--max-rot', dest="MAX_ROT", metavar="max_rotation", \
                    type=int, help="Angle maximum de la rotation dans le jeu d'entraînement (en degrés)", \
                    default=50)
parser.add_argument('-o', '--out-dir', dest="NOM_REP_EXP", metavar="output_dir", type=str, \
                    help="Répertoire où on sauvegarde les fichiers .npz contenant le jeu de données d'entraînement, par défaut : 'exp{numero_experience}'", \
                    default="")
parser.add_argument('-e', '--experience', dest='EXP', metavar='numero_experience', type=int, \
                    help="[1 ou 2] Numéro de l'expérience de l'article de référence pour laquelle on souhaite générer les données d'entraînement", \
                    default="1")
                    
args = parser.parse_args()


# paramètres communs à toutes les expériences
NB_PAIRES_TRAIN = args.NB_PAIRES_TRAIN
                         # pour chaque expérience
TAILLE_IMG = args.TAILLE_IMG
MAX_TRANS = 3            # translation maximale (verticale ou 
                         # horizontale)
MAX_ROT = args.MAX_ROT
NOM_REP_EXP = args.NOM_REP_EXP
NOM_FICH_TRAIN = "train" # nom du fichier contenant les données 
                         # d'entraînement
NOM_FICH_VALID = "valid" # nom du fichier contenant les données 
                         # de validation


if args.EXP != 1 and args.EXP != 2:
    print("Erreur sur le numéro d'expérience")
    exit()
elif NOM_REP_EXP == "":
    NOM_REP_EXP = "exp"+str(args.EXP)

def gen_donnees_exp1():
    """
    Génère les données de l'expérience 1 proposée dans l'article sur 
    les gated autoencoders with tied input weights. 
    """
    def gen_donnees_train():
        """
        Génère les données d'entraînement et les stocke dans un fichier. 
        """
        # génération uniformément aléatoire de NB_PAIRES_TRAIN angles
		# allant de -MAX_ROT à MAX_ROT degrés
        angles = np.floor(np.random.rand(NB_PAIRES_TRAIN) * MAX_ROT * 2 \
						  - MAX_ROT)
        # transx = np.round(np.random.rand(NB_PAIRES_TRAIN) * 7. - 3.5)
        # transy = np.round(np.random.rand(NB_PAIRES_TRAIN) * 7. - 3.5)
        
        # génération des paires d'images
        imgs_x, imgs_y = [], []
        # for tx, ty in zip(transx, transy)
            # x, y = uimg.get_paire_trans(TAILLE_IMG, tx, ty)
        for ang in angles:
            x, y = uimg.get_paire_rot(TAILLE_IMG, ang)
            # mise sous forme vectorielle
            imgs_x.append(x.reshape(-1))
            imgs_y.append(y.reshape(-1))
            
        # regroupement des données dans un dictionnaire
        paires = {"x" : np.array(imgs_x), \
                  "y" : np.array(imgs_y)}
        
        # enregistrement dans un fichier npz
        np.savez(NOM_REP_EXP + "/" + NOM_FICH_TRAIN, **paires)
        
    def gen_donnees_valid():
        """
        Génère les données de validation utilisées pour réaliser
        l'expérience 1 constituées de paires d'images x,y telle que y 
        est le résultat de la rotation de x par chaque angle de -50 à
        50 degrés et ce 2 fois, 0 exclu (on a donc 2 x 100 paires d'images).
        """
        angles = np.arange(-MAX_ROT, MAX_ROT+1.)
        angles = np.delete(angles, MAX_ROT) # suppression de 0
        
        # génération des paires d'images
        imgs_x1, imgs_y1, imgs_x2, imgs_y2 = [], [], [], []
        for ang in angles:
            x1, y1 = uimg.get_paire_rot(TAILLE_IMG, ang)
            x2, y2 = uimg.get_paire_rot(TAILLE_IMG, ang)
            # mise sous forme vectorielle
            imgs_x1.append(x1.reshape(-1))
            imgs_y1.append(y1.reshape(-1))
            imgs_x2.append(x2.reshape(-1))
            imgs_y2.append(y2.reshape(-1))
            
        # regroupement des données dans un dictionnaire
        paires = {"x1" : np.array(imgs_x1), \
                  "y1" : np.array(imgs_y1),
                  "x2" : np.array(imgs_x2), \
                  "y2" : np.array(imgs_y2)}
        
        # enregistrement dans un fichier npz
        np.savez(NOM_REP_EXP + "/" + NOM_FICH_VALID, **paires)
        
    print("Génération des données de l'expérience 1 en cours...")
    # création sur le disque du répertoire stockant les données
    # s'il n'existe pas déjà
    try:
        os.mkdir(NOM_REP_EXP)
    except OSError:
        pass
        
    gen_donnees_train()
    gen_donnees_valid()
    print("Données prêtes à être utilisées. ")
    
def gen_donnees_exp2():
    """
    Génère les données de l'expérience 2 proposée dans l'article sur 
    les gated autoencoders with tied input weights. 
    """
    def gen_donnees_train():
        """
        Génère les données d'entraînement et les stocke dans un fichier. 
        """
        # génération uniformément aléatoire de NB_PAIRES_TRAIN angles
		# allant de -MAX_ROT à MAX_ROT degrés
        angles = np.floor((np.random.rand(NB_PAIRES_TRAIN) * MAX_ROT*1.1 * 2 \
						  - MAX_ROT)/10)*10
        # transx = np.round(np.random.rand(NB_PAIRES_TRAIN) * 7. - 3.5)
        # transy = np.round(np.random.rand(NB_PAIRES_TRAIN) * 7. - 3.5)
        
        # génération des paires d'images
        imgs_x, imgs_y = [], []
        # for tx, ty in zip(transx, transy)
            # x, y = uimg.get_paire_trans(TAILLE_IMG, tx, ty)
        for ang in angles:
            x, y = uimg.get_paire_rot(TAILLE_IMG, ang)
            # mise sous forme vectorielle
            imgs_x.append(x.reshape(-1))
            imgs_y.append(y.reshape(-1))
            
        # regroupement des données dans un dictionnaire
        paires = {"x" : np.array(imgs_x), \
                  "y" : np.array(imgs_y)}
        
        # enregistrement dans un fichier npz
        np.savez(NOM_REP_EXP + "/" + NOM_FICH_TRAIN, **paires)
        
    def gen_donnees_valid():
        """
        Génère les données de validation utilisées pour réaliser
        l'expérience 1 constituées de paires d'images x,y telle que y 
        est le résultat de la rotation de x par chaque angle de -50 à
        50 degrés et ce 2 fois, 0 exclu (on a donc 2 x 100 paires d'images).
        """
        angles = np.arange(-MAX_ROT, MAX_ROT+1.)
        angles = np.delete(angles, MAX_ROT) # suppression de 0
        
        # génération des paires d'images
        imgs_x1, imgs_y1, imgs_x2, imgs_y2 = [], [], [], []
        for ang in angles:
            x1, y1 = uimg.get_paire_rot(TAILLE_IMG, ang)
            x2, y2 = uimg.get_paire_rot(TAILLE_IMG, ang)
            # mise sous forme vectorielle
            imgs_x1.append(x1.reshape(-1))
            imgs_y1.append(y1.reshape(-1))
            imgs_x2.append(x2.reshape(-1))
            imgs_y2.append(y2.reshape(-1))
            
        # regroupement des données dans un dictionnaire
        paires = {"x1" : np.array(imgs_x1), \
                  "y1" : np.array(imgs_y1),
                  "x2" : np.array(imgs_x2), \
                  "y2" : np.array(imgs_y2)}
        
        # enregistrement dans un fichier npz
        np.savez(NOM_REP_EXP + "/" + NOM_FICH_VALID, **paires)
        
    print("Génération des données de l'expérience 2 en cours...")
    # création sur le disque du répertoire stockant les données
    # s'il n'existe pas déjà
    try:
        os.mkdir(NOM_REP_EXP)
    except OSError:
        pass
        
    gen_donnees_train()
    print("Données prêtes à être utilisées. ")
    
def gen_donnees_exp3():
    # TODO
    return
    
def gen_donnees_exp4():
    # TODO
    return
    
if args.EXP == 1:
    gen_donnees_exp1()
if args.EXP == 2:
    gen_donnees_exp2()
# gen_donnees_exp2()
# gen_donnees_exp3()
# gen_donnees_exp4()



