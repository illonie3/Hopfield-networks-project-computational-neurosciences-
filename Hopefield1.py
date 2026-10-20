## import

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd

## fonctions pratiques

M=[
    [1,0,0,0],
    [2,3,0,0],
    [4,8,9,0],
    [8,9,12,34]

]

X=[1,2,7,4,9,10,23,6,8]

def miror(M):
    """ return a symetric matrix by browsing the left down triangle """
    N=len(M)
    for i in range (N):
        for j in range (i+1):
            M[j][i]=M[i][j]

def F(x):
    """ return the activation function of x """
    if x<0:
        return -1
    return 1


## question 1

# 2**N possibilités

## question 2

def random_Hopfield (N):
    """ initiates a random hopfield network with N neurons through a list"""
    X=np.array([0]*N)
    for i in range (N):
        if rd.randint(2) == 1:
            X[i]=-1
        else:
            X[i]=1
    return X


## question 3

def random_poids (N):
    """ initiates a random hopfield poids matrix """
    W=np.zeros((N,N))
    for i in range (N):
        for j in range (i):
            W[i,j]=(rd.random()*2)-1
            miror(W)
    return W

# que signifie un poids négatif ?

## question 4

def square_converter (X):
    """ convert a column matrix into a square matrix """
    N=len(X)
    assert(np.sqrt(N) == int(np.sqrt(N)))
    n=int(np.sqrt(N))
    M=np.zeros((n, n))
    c=0
    for i in range (n):
        for j in range (n):
            M[i,j]=X[c]
            c+=1
    return M


## question 5

def affiche(X):
    """print the network as an image"""
    M=square_converter(X)
    n=len(M)
    im=np.zeros((n, n, 3), dtype=np.uint8)   #pour que le tableau soit en du type "image" ... cf biblio
    fig, ax = plt.subplots()
    for i in range (n):
        for j in range (n):
            if M[i,j]==-1:
                im[i,j]=[255,255,255]
    plt.imshow(im)
    plt.show()

def image(X):
    """return the network as an image matrix"""
    M=square_converter(X)
    n=len(M)
    im=np.zeros((n, n, 3), dtype=np.uint8)
    fig, ax = plt.subplots()
    for i in range (n):
        for j in range (n):
            if M[i,j]==-1:
                im[i,j]=[255,255,255]
    image_display = ax.imshow(im)
    return fig, image_display



## question 6

def network_changing_status(X,W):
    """à partir du statut n du réseau, retourn le statut n+1"""
    N=len(X)
    changing_neuron = rd.randint(N)
    S=0
    for j in range (N):
        S+=W[changing_neuron,j]*X[j]
    X[changing_neuron]=F(S)          #Ainsi, le système peut aussi ne pas changer ... dépendemment de la valeur de F ....


# Ou doit on utiliser le produit matriciel ??? ...... est ce bon ?

## question 7

# je ne comprends pas l'objectif de cette question ... comment quantifier l'évolution du réseau ?


#REVOIR LES FONCTIONNALITES DE MATPLOTLIB

def update_image(img_display, X):    #merci chat gpt de l'aide ... à revoir ...
    """Met à jour l'image affichée avec les nouvelles valeurs de X."""
    M = square_converter(X)
    n = len(M)
    im = np.zeros((n, n, 3), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            if M[i, j] == -1:
                im[i, j] = [255, 255, 255]

    img_display.set_data(im)  # Met à jour l'image


def network_evolution (X,W,delta_t):
    """affiche l'évolution du réseau de neurones pour delta_t itérations"""
    plt.ion()
    fig, image_display = image(X)
    for i in range (delta_t):
        update_image(image_display, X)
        plt.draw()
        plt.pause(0.001)
        network_changing_status(X,W)
    plt.ioff()
    plt.show()


## question 8

# Cf question d'avant ... je crois

## question 9

# Ok, cependant, certains états appartiennent à 2 bassins attracteurs .... cela dépend du neurone choisis aléatoirement lors du premier changement d'état de neurone .... Est ce normal ? Car à ce moment là, les deux bassins attracteurs ne sont pas disjoints ...  ceci est le cas pour l'état X=[-1,1,1,1] pour
#   W=[[ 0.        ,  0.52481755,  0.21594367,  0.34833331],
#       [ 0.52481755,  0.        , -0.90315523,  0.57223559],
#       [ 0.21594367, -0.90315523,  0.        , -0.59151849],
#       [ 0.34833331,  0.57223559, -0.59151849,  0.        ]]

## question 10

def Energy(X):
    """return the energy state of the network"""
    S=0
    n=len(X)
    for i in range(n):
        for j in range (n):
            S+=W[i][j]*X[i]*X[j]
    return -0.5*S

def energy_evolution(X,W,delta_t):
    """return the energy variation during a network evolution"""
    E=[Energy(X)]+[0]*delta_t
    T=range(delta_t+1)
    for i in range (delta_t):
        network_changing_status(X,W)
        E[i+1]=Energy(X)
    plt.plot(T,E)
    plt.show()


## question 11

# faire ca sur papier, ca sera plus simple ...


## question 12

def Hebb_rule (ref): # ref: ensemble de listes de X références
    n=len(ref[0]) # la taille d'une matrice de référence, i.e la taille du réseau
    m=len(ref)  #nombre de références prisent
    W=np.zeros((n,n))
    for i in range (n):
        for j in range (n):
            for s in range (m):
                W[i][j]+=ref[s][i]*ref[s][j]   # les valeurs X[i] et X[j] au s-ème tour d'apprentissage
    np.fill_diagonal(W, 0)
    return W

# J'ai tenté avec votre matrice ref (question 13), cela ne semble marcher que pour de petites matrices ref, et le fait de stocker 3 souvenirs semble deja poser problème : mon dernier élément ref[2] n'est pas attracteur ...

# après quelques recherches, cela pourrait être du à des interférences entre les motifs qui seraient trop proches ... le fait de nomraliser pourrait améliorer ce, mais d'après mes tests ca ne change rien ...

def normalised_Hebb_rule(ref):
    return Hebb_rule(ref)/len(ref)


# après d'autres recherches, ref[2] et ref[1] seraient des points quasi attracteurs...


## question 13

ref=[[1,-1,-1,-1,-1,-1,1,-1,-1,  1,-1,-1,-1,-1,-1,1,-1,-1, -1,1,-1,-1,-1,1,-1,-1,-1,  -1,-1,1,-1,1,-1,-1,-1,-1,   -1,-1,-1,1,-1,-1,-1,-1,-1,  -1,-1,1,-1,1,-1,-1,-1,-1,  -1,1,-1,-1,-1,1,-1,-1,-1,   1,-1,-1,-1,-1,-1,1,-1,-1,  1,-1,-1,-1,-1,-1,1,-1,-1],

[1,-1,-1,-1,-1,-1,-1,-1,-1   ,  1,-1,-1,-1,-1,-1,-1,-1,-1  , 1,-1,-1,-1,-1,-1,-1,-1,-1  ,  1,-1,-1,-1,-1,-1,-1,-1,-1  ,  1,1,1,1,1,-1,-1,-1,-1  ,1,-1,-1,-1,1,-1,-1,-1,-1,    1,-1,-1,-1,1,-1,-1,-1,-1,     1,-1,-1,-1,1,-1,-1,-1,-1,   1,1,1,1,1,-1,-1,-1,-1],

[1,-1,-1,-1,-1,1,-1,-1,-1      ,1,-1,-1,-1,1,-1,-1,-1,-1,   1,-1,-1,1,-1,-1,-1,-1,-1,  1,-1,1,-1,-1,-1,-1,-1,-1,   1,1,-1,-1,-1,-1,-1,-1,-1,           1,-1,1,-1,-1,-1,-1,-1,-1,     1,-1,-1,1,-1,-1,-1,-1,-1,   1,-1,-1,-1,1,-1,-1,-1,-1,    1,-1,-1,-1,-1,1,-1,-1,-1]]   #ne marche pas très bien car motifs proches


essai=[np.array([ 1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
        1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,
       -1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1,  1,
        1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1, -1, -1,
        1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,
    -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1, -1, -1,  1])   , np.array([ 1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,
       -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1,
       -1,  1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1,
        1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1, -1,  1,
        1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1, -1,
       -1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1]),  np.array([ 1,  1,  1, -1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1,
       -1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,
        1, -1,  1, -1, -1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1,  1,
        1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1, -1,
       -1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1, -1,
        1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1])   ,  np.array([ 1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,
        1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,
       -1,  1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,
       -1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,
       -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1,  1,  1])  ,  np.array([ 1,  1,  1,  1,  1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,
        1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1,
        1, -1, -1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1,  1,
       -1, -1, -1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1,
        1,  1,  1,  1,  1,  1, -1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
       -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1])    # marche très bien

]

## question 14

# j'ai bien pu remarquer des points attracteurs opposés surtout pour ref[1] ...







## SUJET 2

import tensorflow as tf

## Question 1

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

## Question 2

def extract(n, image_list, label_list):
    extract_list = []
    for i in range(len(image_list)):
        if label_list[i]==n:
            extract_list.append(image_list[i])
    return np.array(extract_list)

## Question 3

def binarisation(image):
    seuil=75       # car on se situe pour des valeurs de gris entre 0 et 255
    for i in range(len(image)):
        for j in range (len(image[0])):
            if image[i][j] < seuil :
                image[i][j]=1
            else : image[i][j]=-1


## Question 4


def souvenir(n, image_list, label_list):
    category=extract(n, image_list, label_list)
    average = average_image(category)
    binarisation(average)
    return average

def average_image(image_list):
    (n,p)=np.shape(image_list[0])
    resulting_image=np.zeros((n,p))
    for i in range (n):
        for j in range(p):
            for image in image_list :
                resulting_image[i][j]+=image[i][j]
    return resulting_image/len(image_list)

## Question 5

def training(image_list, label_list, *n):   #*n signifie un tuple de n de nombre de valeurs variable
    souvenir_list = []
    label_souvenir_list = []
    for i in range(len(n)):
        souvenir_list.append(souvenir(n[i], image_list, label_list))
        label_souvenir_list.append(n[i])
    print(souvenir_list)
    print(label_souvenir_list)
    W = Hebb_rule(souvenir_list)
    return W,souvenir_list,label_souvenir_list           #PROBLEMES ICI

## Question 6

def category(image, image_list, label_list, *n):
    W , souvenir_list=training(image_list, label_list, *n)
    binarisation(image)
    X=list_converter(image)
    delta_t=500
    network_evolution (X,W,delta_t)
    for souvenir in souvenir_list:
        souvenir = list_converter(souvenir)



def list_converter(matrix):
    X=[]
    for i in range (len(matrix)):
        for j in range (len(matrix[0])):
            X.append(W[i][j])
    return X

def compare (image, souvenir_list):
    for souvenir in souvenir_list:
        if









