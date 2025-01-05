# Sujet de projet AADA - janvier 2024

Ce projet a pour objet de développer un réseau convolutif similaire au réseau appelé U-net capable de segmenter les objets contenus dans une image.

Un algorithme de segmentation permet de détecter chaque objet contenu dans une image en identifiant les pixels qui composent chacun d'entre eux. Le résultat de la segmentation peut être représenté par plusieurs masks, chaque mask étant une image dans laquelle un pixel prend la valeur 1 lorsqu'il appartient à un même type d'objet et la valeur 0 lorsque le pixel n'appartient pas à cet objet.

A titre d'illustration, les deux images suivantes représentent une image couleur acquise dans un environnement routier et le mask des pixels correspondant. Dans ce mask les pixels jaune appartiennent à la classe "signalisation verticale", les pixels vert appartiennent à la végétation.

![Image acquise dans un environnement routier](imageroute.png)

![Illustration d'une segmentation](imagerouteseg.png)

Les labels à retrouver dans les images sont les suivants :

![label](images/label_lyft-udacity-challenge.png)

Le réseau U-net est défini par l'architecture suivante :

![u-net-architecture](u-net-architecture.png)

Dans cette architecture, l'entrée du réseau est à gauche et la sortie à droite. Ainsi en présentant (en mode test) une image à l'entrée du réseau, nous obtenons en sortie plusieurs masks de même résolution que l'image d'entrée (un mask par objet à détecter). L'ensemble des poids est estimé en présentant un batch d'images en entrée du réseau, en imposant la sortie (i.e. les masks des objets à détecter) et en appliquant une méthode d'optimisation (ADAM par exemple).

L'image d'entrée peut etre codée en niveau de gris ou en couleur selon le besoin. La définition des flèches est renseignée sur le schéma précédent.

Le réseau se compose donc de deux parties. La première partie (située a gauche) définie par une succession de couches de convolution, d'activation et de max pooling comme vous l'avez déjà pratiqué lors des séances de TP.
Nous travaillerons sur des images de faibles résolutions. Par conséquent, contrairement à ce qui est indiqué sur l'architecture précédente qui est définie pour les images d'entrée de résolution  572x572, dans notre cas d'usage, les couches successives feront appel respectivement aux nombres de filtres suivants : 16, 16, 32, 32, 64, 64, 128 et 128. Nous obtenons donc 8 feature maps.

La seconde partie du réseau se compose d'une succession de couche de **UpSampling2D**, de convolution, de **concaténation** et d'activation. Aucune couche de max pooling ne compose la seconde partie du réseau. Deux nouveautés apparaissent ici. La couche de **UpSampling2D** et la couche de **concaténation**.
Puisqu'il s'agit d'estimer une image de résolution spatiale identique à l'image d'entrée, il est donc nécessaire d'appliquer les traitements capable d'augmenter la taille des volumes de données (l'inverse des max pooling). Pour cela, nous allons utiliser ces deux nouvelles fonctions. 

Le UpSampling2D est indiquée par la flèche rouge. Elle exploite la fonction ```UpSampling2D()``` de Keras
Pour réaliser ce réseau, la fonction vous est donnée. Elle est appelée ainsi ```d2=pSampling2D(size = (2,2)```. 

Pour obtenir chaque couche, la taille des noyaux de déconvolution est de 3 et leur nombre sera (dans l'ordre d'application) : 64, 64, 32

Pour obtenir chaque couche, la taille des noyaux de convolutions est de 3 et leur nombre sera (dans l'ordre d'application) : 64, 64, 32, 32, 16, 16

La dernière convolution qui permet d'obtenir les différents masks de segmentation est réalisée avec un noyau de 1 et avec un nombre de filtre égal aux nombres de classes.

Attention : après la déconvolution, nous devrez appliquer une couche d'activation (Relu) comme vous le faisiez après une couche de convolution.

Votre travail est d'appeler la fonction déconvolution de la bonne manière i.e. aux bons endroits de l'architecture afin de construire le réseau conformément à ce qui est indiqué dans le schéma de l'architecture précédent.

La fonction de concaténation ```oncatenate([layer1,layes2], axis = 3) ``` est appliquée afin d'assembler les volumes de données de la partie gauche du réseau et de la partie droite. Les couches de la partie gauche du réseau sont celles situées juste avant le max-pooling. Les couches de la partie droite sont celles située après l'appel de la fonction de déconvolution. Les deux couches concaténées requièrent les mêmes résolutions spatiales. Ainsi, cette opération s'effectue selon la troisième composante i.e. la profondeur ```axis=3``` (le nombre de filtres).


## Lecture et conditionnement de données

Voici le script qui vous permet de créer les variables contenant les images qui nous permettront ensuite de lancer l'entraînement du réseau.

```
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
 
from google.colab import drive
drive.mount('/content/drive')

nbr_mask=13
width=160
height=120
 
with open('/content/drive/My Drive/dataset/lyft-udacity-challenge/tabImageMask_all.npy','rb') as f:
  tab_img = np.load(f)
  tab_mask = np.load(f)
 
tab_img=np.array(tab_img)
tab_mask=np.array(tab_mask)
 
train_images, test_images, train_labels, test_labels=train_test_split(tab_img, tab_mask, test_size=.1)

print(tab_mask.shape)
print(tab_img.shape)
# Affichage de la classe 2 de l'image 0
plot.imshow(tab_mask[0][:,:,1])

```


