import numpy as np
import math

def zero_mask(donnees, niv_corrupt):
	"""
	Met à 0 (niv_corrupt * 100) % des donnees de manière aléatoire. 
	:param donnees: la REFERENCE des données à bruiter (numpy array)
	:param niv_corrupt: le niveau de corruption (un pourcentage)
	:returns: les données bruitées
	"""
	NB_DONNEES = donnees.shape[0]
	
	# indices des données rangés aléatoirement
	indices_alea = np.arange(NB_DONNEES)
	np.random.shuffle(indices_alea)
	
	# dont on ne garde que les (niv_corrupt * 100)% premiers
	indices_alea = indices_alea[:math.floor(NB_DONNEES * niv_corrupt)]
	
	# on applique le masque de zéros aux données
	donnees[indices_alea] = 0