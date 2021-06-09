"""
Refs : 
Article Gated Networks : An Inventory
Gated Autoencoders with Tied Input Weights
http://www.cs.toronto.edu/~rfm/code/rae/index.html
"""
import tensorflow as tf
import numpy as np
import math
import os
import time
import scipy.ndimage.filters as scifi
from tensorflow.python import debug as tf_debug

import fonc_denoising as de
import util_affich as aff
import util_mat as umat

class GatedAutoencoderTIW:
    """
    """
    
    # nom du répertoire contenant les sous-répertoires contenant 
    # les modèles sauvegardés
    REP_RESULTATS = "resultats"
    
    # nombre d'exemples de reconstruction à montrer après un entraînement
    NB_EX_MONTRER = 10
    
    def __init__(self, p_type_x, p_type_y, p_taille_img, p_l, \
                 p_n_mappings, p_taux_app, p_niv_corrupt, p_nb_epochs, \
                 p_minibatch_size, p_sigm_filtre_gauss, save_folder=""):
        """
        """
        print("Initialisation du GAE_TIW...")
        self.TYPE_X, self.TYPE_Y = p_type_x, p_type_x
        self.TAILLE_IMG = p_taille_img
        self.NB_PX = self.TAILLE_IMG * self.TAILLE_IMG
        self.L = p_l
        self.N_MAPPINGS = p_n_mappings
        self.TAUX_APP = p_taux_app
        self.NIV_CORRUPT = p_niv_corrupt
        self.NB_EPOCHS = p_nb_epochs
        self.MINIBATCH_SIZE = p_minibatch_size
        self.SIGM_FILTRE_GAUSS = p_sigm_filtre_gauss
        self.SAVE_FOLDER = save_folder
        
        # paires d'images x et y sans et avec corruption
        # que l'on normalise (normalisation par pixel (dim=1))
        # self.x = tf.placeholder(TYPE_X, [NB_PX, None])
        self.y     = tf.placeholder(self.TYPE_Y, [self.NB_PX, None], "Y")
        self.cor_x = tf.placeholder(self.TYPE_X, [self.NB_PX, None], "COR_X")
        self.cor_y = tf.placeholder(self.TYPE_Y, [self.NB_PX, None], "COR_Y")

        # matrices de poids et les biais associés, tous initialisés 
        # aléatoirement
        # rmq : il est nécessaire de travailler en float64 pour gérer les
        # grands nombres (float32 par défaut)
        # INIT ORTHO
        self.ortho = tf.orthogonal_initializer(dtype=tf.float64, seed=time.time())
        self.w1 = tf.Variable(self.ortho(             \
                                [self.N_MAPPINGS, self.L],     \
                                dtype=tf.float64),            \
                              name="w1",                     \
                              dtype=tf.float64)
        self.w2 = tf.Variable(self.ortho([self.N_MAPPINGS, self.L], \
                                          dtype=tf.float64),              \
                              name="w2",                                   \
                              dtype=tf.float64)
        """
        # INIT 10e-3
        self.w1 = tf.Variable(tf.random_uniform(            \
                                [self.N_MAPPINGS, self.L],  \
                                minval=0, maxval=10e-3,     \
                                dtype=tf.float64),          \
                              name="w1", dtype=tf.float64)
        self.w2 = tf.Variable(tf.random_uniform(            \
                                [self.N_MAPPINGS, self.L],  \
                                minval=0, maxval=10e-3,     \
                                dtype=tf.float64),          \
                              name="w2", dtype=tf.float64)
        self.b_mappings = tf.Variable(tf.random_uniform(        \
                                        [self.N_MAPPINGS, 1],   \
                                        minval=0, maxval=10e-3, \
                                        dtype=tf.float64),      \
                                      name="bias_mappings",     \
                                      dtype=tf.float64)
        self.b_output = tf.Variable(tf.random_uniform(          \
                                        [self.NB_PX, 1],        \
                                        minval=0, maxval=10e-3, \
                                        dtype=tf.float64),      \
                                    name="bias_output",         \
                                    dtype=tf.float64)
        """
        
        self.b_mappings = tf.Variable(tf.random_uniform(        \
                                        [self.N_MAPPINGS, 1],   \
                                        minval=0, maxval=10e-3, \
                                        dtype=tf.float64),      \
                                      name="bias_mappings",     \
                                      dtype=tf.float64)
        self.b_output = tf.Variable(tf.random_uniform(          \
                                        [self.NB_PX, 1],        \
                                        minval=0, maxval=10e-3, \
                                        dtype=tf.float64),      \
                                    name="bias_output",         \
                                    dtype=tf.float64)
        
        # eigenvectors de la transformation liant x à y
        # peut être également vu comme une pile de filtres permettant de
        # projeter les entrées sur l'espace factoriel
        # INIT ORTHO
        self.u = tf.Variable(self.ortho(             \
                                [self.NB_PX, self.L],     \
                                dtype=tf.float64),         \
                             name="u",                     \
                             dtype=tf.float64)
        """
        # INIT 10e-3
        self.u = tf.Variable(tf.random_uniform(         \
                                [self.NB_PX, self.L],   \
                                minval=0, maxval=10e-3, \
                                dtype=tf.float64),      \
                             name="u",                  \
                             dtype=tf.float64)
        """

        # construction de la matrice de regroupement de taille L x 2L
        self.p = umat.get_pooling_matrix(self.L).astype(np.float64)

        # construction de la matrice par blocs diagonale B de taille L
        self.b = umat.get_block_diagonal_matrix_b(self.L).astype(np.float64)

        # construction de la matrice de réorganisation R de taille L
        self.r = umat.get_reordering_matrix(self.L).astype(np.float64)

        # construction des matrices de duplication
        self.e1 = tf.concat((np.identity(self.L),  \
                                  np.identity(self.L)), \
                                 axis=0)
        self.e2 = tf.concat((np.identity(self.L),     \
                                 self.b),                 \
                                 axis=0)
        self.e3 = tf.concat((self.r,                     \
                                 np.dot(self.b, self.r)),     \
                                 axis=0)

        # couche factorielles disjointes de dimensions (L, nb_exemples)
        # rmq : constituées de L unités factorielles correspondant aux
        # lignes de la matrice
        self.f_cor_x = tf.matmul(tf.transpose(self.u), self.cor_x)
        self.f_cor_y = tf.matmul(tf.transpose(self.u), self.cor_y)
            
        # couche factorielle unie : contient à la fois les produits inner
        # (partie gauche de la matrice) et les produits cross (partie 
        # droite) ; les matrices constantes utilisées permettent de
        # simuler des multiplications dans l'espace des complexes. 
        # rmq : les valeurs inner fournissent le cosinus de l'angle
        # entre les projections x et y tandis que les valeurs cross
        # en fournissent le sinus (à un facteur multiplicatif près)
        # cf. article pour + de détails
        self.f_cor = tf.matmul(         \
                self.p,                 \
                tf.multiply(            \
                    tf.matmul(          \
                        self.e1,        \
                        self.f_cor_x    \
                    ),                  \
                    tf.matmul(          \
                        self.e2,        \
                        self.f_cor_y    \
                    )                   \
                )                        \
            , name="F_cor") 

        # équations 10
        # mapping layer
        self.m = tf.nn.softplus(tf.add( \
                tf.matmul(              \
                    self.w1,            \
                    self.f_cor          \
                ),                      \
                self.b_mappings         \
            ),name="Mappings")
        # reconstruction
        self.rec = tf.add(                                  \
                tf.matmul(                                     \
                    self.u,                                 \
                    tf.matmul(                              \
                        self.p,                             \
                        tf.multiply(                        \
                            tf.matmul(                      \
                                self.e3,                    \
                                tf.matmul(                  \
                                    tf.transpose(self.w2),  \
                                    self.m                  \
                                ,name="F_out")                           \
                            ),                              \
                            tf.matmul(                      \
                                self.e1,                    \
                                self.f_cor_x                \
                            )                               \
                        )                                   \
                    )                                       \
                ),                                          \
                self.b_output                               \
            ,name="Reconstructed")
            
        tf.summary.scalar("Reconstruct_mean", tf.reduce_mean(self.rec))
        tf.summary.scalar("Reconstruct_mean", tf.reduce_mean(self.rec))

        # la fonction de coût : moyenne de la différence au carré entre
        # la sortie reconstruite depuis des entrées corrompues et la sortie 
        # réelle non corrompue => estimateur de l'erreur de reconstruction
        # On applique également un masque gaussien aux images lors du calcul
        # de l'erreur de reconstruction
        # self.fonc_cout = tf.reduce_mean(tf.pow(self.y - self.rec, 2))
        self.fonc_cout = tf.reduce_mean(tf.pow(\
            scifi.gaussian_filter(self.y, self.SIGM_FILTRE_GAUSS) \
            - scifi.gaussian_filter(self.rec, self.SIGM_FILTRE_GAUSS), 2))
        tf.summary.scalar("Cout",self.fonc_cout)
                
        # optimiseur choisi pour minimiser la fonction de coût
        # GradientDescentOptimizer, RMSPropOptimizer, AdamOptimizer...
        self.optim = tf.train.AdamOptimizer(self.TAUX_APP) \
                     .minimize(self.fonc_cout)
        
        #TENSORBOARD
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("summary")
        
        # ajout des variables et des opérateurs à des collections
        # de sauvegarde pour utilisation lors d'un chargement
        tf.add_to_collection("w1",          self.w1)
        tf.add_to_collection("w2",          self.w2)
        tf.add_to_collection("b_mappings",  self.b_mappings)
        tf.add_to_collection("b_output",    self.b_output)
        tf.add_to_collection("u",           self.u)
        tf.add_to_collection("y",           self.y)
        tf.add_to_collection("cor_x",       self.cor_x)
        tf.add_to_collection("cor_y",       self.cor_y)
        tf.add_to_collection("f_cor_x",     self.f_cor_x)
        tf.add_to_collection("f_cor_y",     self.f_cor_y)
        tf.add_to_collection("f_cor",       self.f_cor)
        tf.add_to_collection("m",           self.m)
        tf.add_to_collection("rec",         self.rec)
        tf.add_to_collection("fonc_cout",   self.fonc_cout)
        tf.add_to_collection("optim",       self.optim)
        
        # opération de sauvegarde du modèle
        self.saver = tf.train.Saver()

        # définition de l'initialisation des variables
        self.init = tf.global_variables_initializer()
        print("GAE_TIW prêt à l'entraînement. ")
        
    def apprendre(self, donnees):
        """
        """
        donnees_x, donnees_y = donnees["x"], donnees["y"]
        
        # normalisation des images pour chaque pixel
        moy_px_x = donnees_x.mean(0)
        std_px_x = donnees_x.std(0)
        donnees_x = (donnees_x - moy_px_x) / std_px_x
        donnees_y = (donnees_y - moy_px_x) / std_px_x
        
        # transposée pour obtenir les images en colonne
        donnees_x, donnees_y = donnees_x.T, donnees_y.T
        
        fonc_cout_vals = []

        with tf.Session() as sess:
            #Commenter la prochaine ligne pour desactiver le debugging
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            
            print("Entrainement du GAE...")
            
            # corruption des images
            cor_donnees_x = donnees_x.copy()
            cor_donnees_y = donnees_y.copy()
            for i in range(cor_donnees_x.shape[1]):
                de.zero_mask(cor_donnees_x[:,i], self.NIV_CORRUPT)
                de.zero_mask(cor_donnees_y[:,i], self.NIV_CORRUPT)
            
            # exécution de la phase d'initialisation des variables
            sess.run(self.init)
            self.writer.add_graph(sess.graph)
            # print(sess.run([w1]))
            # print(sess.run([w2]))
            # print(sess.run([bias_mappings]))
            # print(sess.run([bias_output]))
            # print(sess.run([u]))
            
            NB_EXEMPLES = donnees_x.shape[1]
            global_step = 0    
            # entraînement EPOCHS fois
            for i in range(self.NB_EPOCHS):
                startTimer = time.time()
                print("Entrainement",(i+1),"/",self.NB_EPOCHS)
                
                # indices des exemples rangés aléatoirement
                indices_alea = np.arange(NB_EXEMPLES)
                np.random.shuffle(indices_alea)
                
                # sélection des MINIBATCH_SIZE 1ers exemples
                i = 0
                j = self.MINIBATCH_SIZE
                
                # tant qu'il reste des exemples pour l'entraînement courant
                coutMoy = 0
                while j < NB_EXEMPLES:
                    # on récupère les lots aléatoires courants
                    lot_cor_x = cor_donnees_x[:,indices_alea[i:j]]
                    lot_cor_y = cor_donnees_y[:,indices_alea[i:j]]
                    lot_y = donnees_y[:,indices_alea[i:j]]
                    i += self.MINIBATCH_SIZE
                    j += self.MINIBATCH_SIZE
                    # puis on lance la descente de gradient sur le lot
                    # (on récupère également le coût)
                    with tf.device("/cpu:0"): # optim pour calcul
                        _, cout, summary = \
                            sess.run([self.optim, self.fonc_cout, self.merged], \
                                feed_dict={self.cor_x: lot_cor_x, \
                                           self.cor_y: lot_cor_y, \
                                           self.y:     lot_y})
                        global_step+=1
                        self.writer.add_summary(summary,global_step)
                        coutMoy += cout
                coutMoy /= (NB_EXEMPLES / self.MINIBATCH_SIZE)
                    
                # print(sess.run([w1]))
                # print(sess.run([w2]))
                # print(sess.run([bias_mappings]))
                # print(sess.run([bias_output]))
                # print(sess.run([u]))
                print("Cout moyen du dernier entrainement :",coutMoy)
                print("Duree du dernier epoch :",time.time()-startTimer)
                fonc_cout_vals.append(coutMoy)
            
            print("Entrainement terminé.")
            # on créé un répertoire "resultats" si celui-ci n'existe pas
            try:
                os.mkdir(self.REP_RESULTATS)
            except OSError:
                pass
            
            # création sur le disque du répertoire stockant le modèle créé
            if self.SAVE_FOLDER == "":
                nom_rep_modele = self.get_nom_rep_modele()
            else:
                nom_rep_modele = self.SAVE_FOLDER
            try:
                os.mkdir(nom_rep_modele)
            except OSError:
                pass
                
            # sauvegarde du modèle
            self.saver.save(sess, nom_rep_modele + "/modele")
            print("Le modèle créé a été sauvegardé. ")
            
            # pour affichage de NB_EX_MONTRER images reconstruites
            img_predites = sess.run(                                       \
                    self.rec,                                             \
                    feed_dict={                                           \
                        self.cor_x: donnees_x[:,:self.NB_EX_MONTRER], \
                        self.cor_y: donnees_y[:,:self.NB_EX_MONTRER]})
        
        aff.sauvegarderFoncCout(fonc_cout_vals,             \
                                nom_rep_modele + "/cout",     \
                                "png")
        f = open(nom_rep_modele + "/cout.txt", "w")
        for val in fonc_cout_vals:
            f.write("%s\n" % val)
        f.close()
        print("Un historique de la fonction de coût a été sauvegardé. ")
        
        # formatage pour affichage
        # tranposition
        donnees_y    = donnees_y.T
        img_predites = img_predites.T
        # "dénormalisation"
        donnees_y    = (donnees_y * std_px_x) + moy_px_x
        img_predites = (img_predites * std_px_x) + moy_px_x
        
        img_attendues = aff.multiVecToMultiMat(             \
                            donnees_y[:self.NB_EX_MONTRER],     \
                            self.TAILLE_IMG, self.TAILLE_IMG)
        img_predites = aff.multiVecToMultiMat(         \
                            img_predites,         \
                            self.TAILLE_IMG, self.TAILLE_IMG)
        
        aff.sauvegarderGrilleImages2L(                     \
            self.NB_EX_MONTRER, img_attendues, img_predites, \
            nom_rep_modele + "/attendu_vs_predit", "png")
        print("Des exemples de reconstruction ont été sauvegardés. ")
        
        return fonc_cout_vals
        
    def get_nom_rep_modele(self):
        """
        :returns: le nom du répertoire où sont stockés les résultats pour 
                  les paramètres courants (!= du répertoire REP_RESULTATS
                  qui est le parent de ce répertoire).
        """
        return (self.REP_RESULTATS + "/" + \
                "L" + \
                str(self.L) + "_" + \
                "map" + \
                str(self.N_MAPPINGS) + "_" + \
                "epochs" + \
                str(self.NB_EPOCHS) + "_" + \
                "tlot" + \
                str(self.MINIBATCH_SIZE) + "_"+ \
                "corrupt" + \
                str(self.NIV_CORRUPT))
