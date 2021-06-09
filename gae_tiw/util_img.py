import numpy as np
import matplotlib.pylab as plt
import cv2

def img_alea_norm(taille):
    """
    Génère une image carrée de manière aléatoire suivant la distribution
    normale et dont la taille est passée en paramètre. 
    Valeur des pixels de 0 à 255 (inclus).
    
    :param taille: la longueur en pixels d'un côté de l'image
    :returns: le vecteur numpy représentant l'image
    """
    # images [0 ; 255]
    return (np.floor(np.minimum(np.maximum(np.random.normal(size=taille*taille, loc=128, scale=64),0.),255.))).reshape(taille, taille)
    # images binaires
    # return (np.round(np.random.normal(size=taille*taille, loc=0.5, scale=0.25)) * 255.).reshape(taille, taille)

def affich_img(img):
    """
    Affiche une image carrée représentée par un vecteur numpy de taille
    paire. 
    
    :param img: la matrice numpy image
    """
    plt.figure()
    plt.imshow(img, cmap='plasma')
    plt.axis("off")
    plt.show()
    
def affich_imgs_2L(imgs_x, imgs_y):
    """
    Affiche les images x sur la première ligne et les images y
    sur la seconde ligne. 
    
    :param img_x: les images à afficher sur la première ligne
    :param img_y: les images à afficher sur la seconde ligne
    :throws ValueError: si le nombre d'images à afficher n'est pas le même
                        en 1ère et 2ème ligne
    """
    if len(imgs_x) != len(imgs_y):
        raise ValueError("ERREUR - affich_imgs_2L() : il doit y avoir "
                       + "le même nombre d'images en 1ère et 2ème lignes.")
    
    plt.figure()
    # décomposition de la figure en une grille de 2 lignes
    fig, grille = plt.subplots(2, len(imgs_x))
    
    for i in range(len(imgs_x)):
        grille[0][i].imshow(imgs_x[i], cmap='plasma')
        grille[0][i].axis("off")
        grille[1][i].imshow(imgs_y[i], cmap='plasma')
        grille[1][i].axis("off")

    plt.show()

def get_transfo(img, mat_transfo):
    """
    Transforme l'image selon la matrice de transformation fournie.
    Donne une valeur aléatoire suivant une loi normale aux pixels 
    non couverts.
    
    :param img: la matrice numpy image à transformer
    :returns: l'image transformée
    """
    # application de la translation sur l'image
    # les pixels non couverts prennent temporairement la valeur -1
    # INTER_NEAREST : choix d'un voisin proche lors d'une interpolation
    # vs INTER_LINEAR création d'une valeur
    nouv_img = np.floor(cv2.warpAffine(img, mat_transfo, (img.shape[0], img.shape[0]), borderValue=-1,flags=cv2.INTER_NEAREST))
    
    # récupére les indices des pixels non couverts
    ind_ncouv = np.where(nouv_img == -1)
    
    # génére aléatoirement les pixels non couverts selon une loi normale
    # images [0 ; 255]
    nouv_img[ind_ncouv] = np.floor(np.minimum(np.maximum(np.random.normal(size=ind_ncouv[0].size, loc=128, scale=64),0.),255.))
    
    # images binaires
    # nouv_img[ind_ncouv] = np.round(np.random.normal(size=ind_ncouv[0].size, loc=0.5, scale=0.25)) * 255.
    
    return nouv_img

def translater(img, nb_px_x, nb_px_y):
    """
    Translate une image d'un certain nombre de pixels en abscisse 
    et en ordonnée. Les pixels qui ne sont plus couverts par l'image
    de base sont alors choisis aléatoirement.
    
    :param img: la matrice numpy image
    :param nb_px_x: le nombre de pixels de translation en abscisse
    :param nb_px_y: le nombre de pixels de translation en ordonnée
    :returns: l'image translatée
    """
    # matrice de translation
    mat_trans = np.float32([[1,0,nb_px_x],[0,1,nb_px_y]])
    
    return get_transfo(img, mat_trans)

def pivoter(img, angle):
    """
    Effectue une rotation sur une image d'un certain angle. 
    Les pixels qui ne sont plus couverts par l'image de base sont alors 
    choisis aléatoirement.
    
    :param img: la matrice numpy image
    :param angle: l'angle de rotation en degrés
    :returns: l'image pivotée
    """
    # indices centre de l'image
    centre_img = tuple(np.array(img.shape)/2.)
    
    # matrice de rotation
    mat_rot = cv2.getRotationMatrix2D(centre_img, angle, 1.)
    
    return get_transfo(img, mat_rot)

def get_paire_trans(taille, nb_px_x, nb_px_y):
    """
    Construit une paire d'images telle que la seconde est obtenue 
    à partir d'une translation de la première. 
    
    :param taille: la longueur en pixels du côté d'une image
    :param nb_px_x: le nombre de pixels de translation en abscisse
    :param nb_px_y: le nombre de pixels de translation en ordonnée
    :returns: la paire d'image générée dans un tuple
    """
    img_x = img_alea_norm(taille)
    img_y = translater(img_x, nb_px_x, nb_px_y)
    return (img_x, img_y)

def get_paire_rot(taille, angle):
    """
    Construit une paire d'images telle que la seconde est obtenue 
    à partir d'une rotation de la première. 
    
    :param taille: la longueur en pixels du côté d'une image
    :param angle: l'angle de rotation en degrés
    :returns: la paire d'image générée dans un tuple
    """
    img_x = img_alea_norm(taille)
    img_y = pivoter(img_x, angle)
    return (img_x, img_y)

# exemples de tests
# imgs_x, imgs_y = [], []
# for i in range(10):
    # x, y = get_paire_trans(13, i, i)
    # imgs_x.append(x)
    # imgs_y.append(y)
# affich_imgs_2L(imgs_x, imgs_y)

# print(imgs_x[0])
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(imgs_y[0])
# print("##########################")
# print(imgs_x[1])
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(imgs_y[1])

# imgs_x, imgs_y = [], []
# for i in range(10):
    # x, y = get_paire_rot(13, i*10)
    # imgs_x.append(x)
    # imgs_y.append(y)
# affich_imgs_2L(imgs_x, imgs_y)

# print(imgs_x[0])
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(imgs_y[0])
# print("##########################")
# print(imgs_x[1])
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(imgs_y[1])
