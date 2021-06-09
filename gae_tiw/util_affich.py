import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
 
def sauvegarderFoncCout(foncCoutVal, cheminFichier, extension):
    """
    Affiche les valeurs prises par la fonction de coût en fonction du nombre
    d'itérations. 
    
    :param foncCoutVal: la liste des valeurs prises par la fonction de coût
    """
    fig = plt.figure()
    plt.plot(range(1,len(foncCoutVal)+1), foncCoutVal)
    plt.xlabel("Nombre d'entraînements (epochs)")
    plt.ylabel("Valeur de la fonction de coût")
    plt.title("Evolution de la fonction de coût")
    fig.savefig(cheminFichier + "." + extension)

def sauvegarderGrilleImages2L(nbColonnes, imagesL1, imagesL2, \
							  cheminFichier, extension):
    """
    Affiche des images de taille longImg * hautImg dans une grille de taille
    2 x nbColonnes. 
    
    :param nbColonnes: nombre de colonnes de la grille à afficher
    :param   imagesL1: les images à afficher en ligne 1
    :param   imagesL2: les images à afficher en ligne 2
    """
    plt.figure()
    # décomposition de la figure en une grille 2 * nbColonnes
    fig, grille = plt.subplots(2, nbColonnes)

    # pour chaque colonne
    for i in range(nbColonnes):
        grille[0][i].imshow(imagesL1[i], cmap='plasma')
        grille[0][i].axis("off")
        grille[1][i].imshow(imagesL2[i], cmap='plasma')
        grille[1][i].axis("off")

    fig.savefig(cheminFichier + "." + extension)

def multiVecToMultiMat(multiVec, nbLignes, nbColonnes):
    """
    Transforme une liste de vecteurs en une liste de matrices de taille
    nbLignes * nbColonnes
    
    :param   multiVec: la liste de vecteurs à transformer
    :param   nbLignes: le nombre de lignes de la matrice résultat
    :param nbColonnes: le nombre de colonnes de la matrice résultat
    """
    return [np.reshape(vecImg, (int(nbLignes),int(nbColonnes))) for vecImg in multiVec]

