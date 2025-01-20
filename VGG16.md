# VGG-16 from sratch to fine tunning for image classification

Dans ce sujet il s'agit d'implanter, d'entraîner et d'évaluer un réseau appelé VGG-16 pour une tâche de classification.
L'architecture de ce réseau est la suivante :

![image](images/VGG-16.png)

Cette architecture est composée d'une première partie convolutive qui représente l'extraction des caractéristiques et d'une
second partie totalement connectée qui assure la tâche de classification. Le schéma précédent indique pour chaque layer, le nombre de filtres ou de neurones,
le type de padding et les couches de pooling. proposés par K. Simonyan et A. Zisserman de l’université d’Oxford. Grâce à cette architecture, les auteurs ont 
gagné la compétition ILSVRC (ImageNet Large Scale Visual Recognition Challenge) en 2014 en atteignant une précision de 92.7% sur la base d'images Imagenet.
La particularité de ce réseau est qu'il est le premier à avoir proposé l'usage de noyaux de convolution de plus petites dimensions (3×3).
Pour rappell, la base d'images ImageNet est composée de 14 millions d'images classées en 1000 classes. Ce sont des images RGB de résolution 224x224.

Dans ce sujet, nous présenterons deux manières d'entraîner cette architercture de réseau. La première est dite 'training from scratch' i.e. à partir d'un réseau initialisé
avec des poids et des paramètres de filtres aléatoires. La seconde fait appel à une méthode dite de 'fine tuning'. Nous utiliserons la base d'images CIFAR-100.

Dans ce sujet, vous serez moins guidés que dans les sujets précédents. Seules les nouveautés ou les parties de script un peu délicats vous seront présentées. 
Il s'agit pour vous d'aller rechercher les éléments rencontrés dans les 3 sujets précédents pour réaliser ce travail. Par ailleurs, à vous désormais d'importer les librairies
requises pour développer votre script.

## Entrainement 'from scratch'

### Etape 1 : Définition du modèle VGG-16

A partir du schéma de l'architcture VGG-16 ...

**Question : définir le modèle VGG-16 de référence à l'aide des instructions Keras dont vous avez désormais connaissance. Afficher le summary de l'architcture.**

Cette défintion sera le modèle de référence i.e. le modèle que nous ferons évoluer afin d'atteindre une défintion optimale.

**Question : charger les images de la base d'image CIFAR-10 et créer un sous-ensemble de validation de 20% de la base de training.**


**Question : lancer l'apprentissage du réseau. Choisissez l'une des fonctions de loss que nous avons utiliser jusqu'à maintenant
et modifier les labels en conséquence. Pour cette question, nous testerons les optimizers Adam et SGD. Nous fixerons le learning_rate=0.1,
un batch_size=64 et un nombre d'epoch=10. Pour ces deux optimizers, comparer les valeurs obtenues en loss, accuracy, val_loss et val_accuracy.**

Nous allons maintenant modifier l'architecture du réseau en deux temps. Tout d'abord nous allons ajouter des couche de BatchNormalization
avec la fonction ```model.add(BatchNormalization())```. Cette fonction permet de normaliser les valeurs fournies par la fonction d'activation 
sur l'ensemble d'un batch. 

**Question : ajouter une couche de BatchNormalization après chaque couche de convolution et afficher le summary de l'architcture. Quels sont les changements 
portés par cet ajout par rappor au réseau précédent ?**

**Question : relancer l'apprentissage pour les deux optimizers. Quels constats pouvons nous faire en comparant à nouveau les valuers de loss, d'accuracy, de val_loss et de val_accuracy ?**

**Question : ajouter des couche de dropout avec un taux d'extinction de 0.2 après chaque couche de convolution mais jamais avant ou après les couches de pooling. Relancer 
l'apprentissage pour les deux optimizers. Quels constats pouvons nous faire en comparant à nouveau les valuers de loss, d'accuracy, de val_loss et de val_accuracy ?**

### Etape 2 : entraînement du réseau "optimal"

**Question : lancer l'entraînement du réseau que nous aurons qualifié d'"optimal" sur un nombre d'epoch=20 et enregistrer sur votre drive les paramètres du réseau et son architecture.**

**Question : évaluer ce réseau sur la base de test et afficher la matrice de confusion.**

## Fine-tuning d'un VGG-16 pré-entraîné

Comme nous l’avons déjà mentionné cours, le temps d’entraînement d’un modèle comme VGG peut être très long surtout si les ressources matériels sont limitées. 
VGG, comme beaucoup d'autres réseaux, ont été entraînés sur la base ImageNet pour cette tâche de classification. Il est donc intéressant de récupérer les poids du modèle
déjà entraîné et notamment les filtres dans les couches de convolution afin de profiter de la capacité de "projection" de ces couches. Ainsi, la phase d'entraînement que 
nous souhaitons mettre en place pour assurer une nouvelle tâche de classification (ici sur les 10 classes CIFAR) se réduit à l'apprentissage des poids des couches 
totalement connectées du MLP et éventuellement de quelques couches de la partie convolutive (des dernières couches).

Keras met à disposition un grand nombre d'architectures neuronales qui ont été pré-entraînées sur ImageNet (https://keras.io/api/applications/). Nous allons donc en profiter et
récupérer les poids et les paramètres d'un VGG-16 que nous adapterons à la tâche de classification sur CIFAR-10.

Voici les lignes de codes pour récupérer le VGG-16 pré-entraîné sur ImageNet :
```
import keras
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras import backend as K
from keras.applications.vgg16 import VGG16

# Modèle VGG16
base_model = VGG16(weights='imagenet', include_top=False,input_shape=(32,32,3))
```

Un fois la structure base récupérée, il faut "freezer" c'est-à-dire figer les couches du réseau que nous ne souhaitons pas modifier pendant la phase d'entraînement :

```
# Freezer les couches du VGG16
for layer in base_model.layers: 
    layer.trainable = False
```

Attention, il faut noter que lors du chargement du VGG, nous demandons à ne pas récupéer les dernière couches totalement connectées, uniquement les couches de convolution.

Par conséquent, nous allons ajouter la couche de neurones totalement connectés comme nous l'avons fait dans l'architecture du réseau de la première partie. La synthaxe Keras de définition
du réseau est différente de ce que nous avons vu dans les sujets précédents. C'est une manière d'introduire une définition alternative qui permet de "nommer" les couches, apportant ainsi une certaine
souplesse dans le définition du réseau. Cela vous sera très utile lorsque vous aurez à définir des architectures non linéaires, multi-entrées avec des bouclages et des récurrences. Voici les lignes de code :

```
# Définition du shape de l'entrée du réseau
input = Input(shape=(32,32, 3),name = 'image_input')
output_vgg16_pretrained = base_model(input)

# Couche de flatten
x = Flatten(name='flatten')(output_vgg16_pretrained)

# Couche dense de 512 neurones qui prend en entrée la sortie x de la couche flatten
x = Dense(512, activation='relu', name='fc1')(x)

# Couche de sortie de 10 neurones qui prend en entrée la sortie x de la couche cachées fc1
x = Dense(10, activation='softmax', name='predictions')(x)

# Création du modèle
my_model = Model(inputs=input, outputs=x)

# Affichage
my_model.summary()

```

Il nous reste désormais à lancer l'entraînement avec les codes que vous connaissez....

**Question : lancer l'entraînement avec learning_rate=0.1, epoch=15, batch_size=64 sur la base de training et la base de validation. Que constatons nous ?**

**Question : lancer à nouveau en réduisant le learning_rate=0.01 et en augmentant le nomre d'epoch=100. Afficher les courbes d'apprentissage. Faite l'évaluation sur la base de test et
afficher la matrice de confusion.**

Comme vous pourrez le constater, les performances sur la base d'entraînement et la base de validation ne sont pas à la heuteur de ce qu'on pourrait attendre. Afin d'améliorer les résultats, une solution est de normaliser les bases d'entraînement. Pour cela, vous pouvez utiliser la fonction suivante :
```
def normalize(X_train,X_test):
    # normalise les entrées pour obtenir une valeur moyenne de 0 et une variance unitaire
    # en fonction de la statistique des données
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print(mean)
    print(std)
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test
```

Appliquer la fonction normalize pour normaliser vos données d'apprentissage et de test :
```
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = normalize(x_train, x_test)
```

## Pour aller plus loin

**Question : jouer avec un autre réseau pré-entraîné disponible.**

**Question : jouer avec la dataset CIFAR-100. Ce jeu de données comporte 100 classes différentes, chaque classe comportant 500 images d'entraînement et 100 images de test.**






