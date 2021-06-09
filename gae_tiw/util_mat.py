import numpy as np
import scipy.linalg
import math

def get_block_diagonal_matrix(nb_blocs, bloc):
    if nb_blocs <= 0:
        return np.array([])

    mat = bloc
    for i in range(nb_blocs-1):
        mat = scipy.linalg.block_diag(mat, bloc)
    return np.array(mat)

def get_pooling_matrix(L):
    # construction par blocs
    return get_block_diagonal_matrix(L, [1,1])

def get_block_diagonal_matrix_b(L):
    # construction par blocs
    b = get_block_diagonal_matrix(math.floor(L / 2), [[0,1],[-1,0]])
    # si L est impair => ne reste plus du bloc final qu'un 0
    if L > 0 and L % 2 != 0:
        b = scipy.linalg.block_diag(b, [0])
    return b

def get_reordering_matrix(L):
    # cas particuliers
    if L <= 0:
        return np.array([])
    elif L == 1:
        return np.array([1])

    # resp. partie gauche et droite de r
    r_part1 = get_block_diagonal_matrix(math.floor(L / 2), [[1],[0]])
    r_part2 = get_block_diagonal_matrix(math.floor(L / 2), [[0],[-1]])
    
    r = []
    # si L est impair
    if L % 2 != 0:
        # alors on ajoute une colonne de zéros entre les 2 parties de r...
        r = np.concatenate((r_part1, np.zeros((L-1,1))), axis=1)
        r = np.concatenate((r, r_part2), axis=1)
        # puis une ligne de 0 avec un 1 en son centre
        ligne_f = np.zeros((1,L))
        ligne_f[0][math.floor(L/2)] = 1.
        r = np.concatenate((r, ligne_f), axis=0)
    else:
        # sinon on concatène simplement les 2 parties de r ensemble
        r = np.concatenate((r_part1, r_part2), axis=1)
    
    return r