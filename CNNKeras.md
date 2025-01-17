# Mon premier réseau de neurones convolutifs en Keras

Dans les deux sujets précédents vous avez appris à définir, entraîner, évaluer et inférer un réseau de neurones totalement connectés (MLP) que vous
avez appliqué pour assurer une tâche de regression logistique (sur le jeu de données Iris) et de classification multiclasse (sur 
les jeux de données Iris et MNIST). Vous avez également connaissance de quelques commandes pour afficher les courbes d'apprentissage, de déterminer
la qualité de l'entraînement, l'apparition de surapprentissage et pour sauvegarder les poids et la structure du réseau pour un usage ultérieure.

Dans ce sujet, nous nous plaçons dans le cadre d'une application de classifications d'images et nous allons apprendre à définir l'architecture d'un réseau de neurones convolutifs (CNN) et à l'entraîner.
Nous commencerons par faire ce travail sur le jeu de données MNIST. Puis nous l'étendrons à la base d'images couleur CIFRAR-10 .

## Chargement et préparation des données

**Question : commencer par importer les librairies donc vous aurez besoin puis charger les images MNIST comme indiqué dans le sujet précédent. Le jeu d'entraînement avec ses labels
seront chargés dans les variables x_train, y_train. Le jeu de test avec ses labels seront chargés dans les variables x_test, y_test.**

Puis nous allons afficher 25 images du jeu de données d'entrainement avec ces quelques lignes de code :

```
# MNIST
class_names = ['ZERO', 'UN', 'DEUX', 'TROIS', 'QUATRE', 'CINQ',
               'SIX', 'SEPT', 'HUIT', 'NEUF']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i]])
plt.show()
```

**Question : faire de même avec l'ensemble des images de tests**

## Spécification du modèle convolutif

Un réseau de neurones convolutifs de base est structuré en deux parties. La premmière partie du réseau réalise la projection des données qui lui sont placées en entrée, dans un espace de "caractéristiques" (feature map) : elle fournit ainsi un vecteur d'informations qui "résume" l'information contenue dans les données d'entrée au regard d'une tâche à réaliser (une classification d'images par exemple). La second partie du réseau utilise un réseau de neurones complètement connecté (MLP) pour réaliser la tâche proprement dite à partir de ce vecteur de caractéristiques. Dans ce premier exemple, nous allons définir un CNN qui sera composé d'une seule couche de convolution et d'une couche de __pooling__. La couche de convolution sera composée de 32 filtres (32 noyaux ou kernels) de taille 3x3; elle fera appel à la fonction ```layers.Conv2D()``` et à une fonction d'activation de type Relu. La couche de __pooling__ fera appel à l'opérateur ```MaxPooling2D()``` sur un voisinage 2x2. Voici les lignes Keras qui définissent l'architecture neuronale, de manière similaire à ce que nous avons fait précédemment :

```
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Couche "feature map"
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Couche classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

```
La couche de convolution 2D prend en entrée une image de taille 28x28 en niveau de gris (input_shape=(28,28,1)).
Sachant que par défaut le paramètre __stride = (1,1)__ et le paramaètre __padding='valid'__ sont passés à la fonction Conv2D, la sortie de cette couche est de taille 26x26x32, 32 correspondant au nombre de filtres.
__'valid'__ signifie qu'aucun padding n'est appliqué, donc les effets de bord de la convolution subsistent et entrainent une baisse de la résolution.

La couche de pooling (2,2) réduit la résolution par 2 dans chaque dimension en replaçant la valeur d'un pixelv (x,y) par la valeur maximale d'un voisinage 2x2 centré sur (x,y). Par défaut le paramètre de __padding='valide'__ ce qui fait que la sortie de cette couche est de résolution (13x13x32).

Dans la second partie du réseau, nous retrouvons la couche __Flatten__ qui permet d'applatir la sortie de la couche précédente.

**Question : quelle est la taille de la sortie de la couche __Flatten__?**

**Question : quel est le nombre de paramètres du modèle complet ?**

La valeurs des filtres sont initialisées de manière aléatoire. La phase d'entraînement va permettre de les déterminer au regard d'une fonction de classification d'images assurée sur la base MNIST.

## Entraînement du modèle

L'objectif est de déterminer les poids du modèle i.e. les valeurs des filtres et les poids de la partie totalement connectée.

**Question : sur la base de ce que vous avez fait à la fin du TP précédent, écrire les lignes de code pour définir les paramètres de l'apprentissage et pour lancer le fitting du modèle sur 10 epochs.**

Attention, nous avons une base d'apprentissage (X_train, y_train) et une base de test (X_test,y_test). Seule la base X_train devra être utilisée pour l'entraînement. Donc nous devons en extraire une base de validation (X_val, y_val) que nous poasserons à la fonction ```model.fit()```.
Toujours comme dans la fin du TP précédent, nous utiliserons la fonction de loss __CategoricalCrossentropy__ : nous aurons besoin de transformer les labels des échantillons de la base d'apprentissage et de la base de test comme nous l'avons déjà fait dans le sujet précédent.

## Evaluation et prédiction

**Question : évaluer le modèle obtenu sur la base de test complète et calculer l'accuracy.**

**Question : retrouver les images de la base de test sur lesquelles le modèle fait de mauvaises prédictions.**

## Complexification du modèle

A partir de cette partie lancer un environnement colab avec GPU car les modèles comporteront plus de paramètres estimés.

**Question : définir un CNN dont la composition est la suivante et donner son nombre de paramètres :**

```
couche CNN_1 : 32 filtres avec stride=(1,1) et padding='same'
couche CNN_2 : 16 filtres avec stride=(1,1) et padding='same'
couche CNN_3 : 8 filtres avec stride=(1,1) et padding='same'

et

couche FC cachée : 64 neurones
couche de sortie : 10 neurones**
```
**Question : lancer son entraînement sur la base d'entrainement sur 10 épochs, des batchs de 32 images et un split de 20% pour la base d'évaluation.**

**Question : evaluer ce modèle sur la base de test et calculer la valeur d'accuracy.**


# Application d'un CNN à un jeu d'images en couleur

Nous allons désormais définir un CNN capable d'opérer une tâche de classification d'images en couleur.
Pour cela nous allons utiliser la base d'images CIFAR-10. Cet ensemble CIFAR-10 contient 60 000 images couleur de dimension 32 X 32 en 3 canaux répartis en 10 classes. Les données d'entraînement contiennent 50 000 images et les données de test 10 000. (https://www.cs.toronto.edu/~kriz/cifar.html). Il s'agit d'un problème multi-classes avec 10 étiquettes. Les données sont réparties de manière équilibrée entre les étiquettes. Les étiquettes sont les suivantes :
```
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```

Le chargement et la normalisation des images s'effectue de la même manière que pour MNIST :

```
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalisation des valeurs RGBdes pixels
x_train, x_test = x_train / 255.0, x_test / 255.0

```

## Définition, entraînement et évaluation

**Question : définir, entraîner, tracer les courbes et évaluer l'architecture convolutive suivante. Utiliser l'algorithme Adam, fixer le nombre d'epochs à 100 et la taille des batchs à 32 images. Garder à l'esprit que les images sont RGB donc certaines parties du code précédent devront être modifiées en conséquence. Vous mènerez l'évaluation sur la base de test. Voici l'architecture :**

```
couche CNN_1 : 256 filtres avec stride=(1,1) et padding='same'
couche CNN_2 : 128 filtres avec stride=(1,1) et padding='same'
couche CNN_3 : 64 filtres avec stride=(1,1) et padding='same'
couche CNN_3 : 32 filtres avec stride=(1,1) et padding='same'

et

couche FC cachée : 64 neurones
couche de sortie : 10 neurones
```
**Question : quelles constatations pouvions nous en tirer si les couches CNN utilise un padding='valid' ?**

**Question : en analysant les courbes d'apprentissage que constatez-vous (dans le cas padding='same') ?**

Pour remédier à ce problème plusieurs solutions existent dont
1. l augmentation du nombre de données (data augmentation)
2. l'ajout de couches de __dropout__
3. la réduction de la complexité du réseau
4. etc.

Nous allons tester l'ajout de couches de __dropout__ par ```model.add(layers.Dropout(0.25))```
Nous le ferons après chaque couche de pooling. La valeur passée à la fonction Dropout() est la proportion
de paramètres qui n'est pas mise à jour pendant l'entraînement. Dans notre exemple, 25% des paramètres ne sont
pas mis à jour. Ces 25% sont tirés de manière aléatoire.

**Question : définir cette structure de réseau et entraîner le réseau sur 50 epochs. Comparer les courbes d'apprentissage de cette expérimentation avec les courbes obtenues 
sans __dropout__**

## Analyse des résultats

Nous allons étudier plus précisément les résultats donnés par le réseau appris sur la base de test. Dans un premier temps, nous allons afficher les 10 premières erreurs de prédiction pour chacune des classes CIFAR-10. Une telle fonction est disponible dans la librairie pml_utils que vous pouvez télécharger du dossier ```tools``` de ce dépot dans le dossier dans lequel votre notebook actif est enregistré.

Voici les lignes de code (ici pour la classe "truck") :

```
import os
os.chdir("nom du dossier dans lequel est sauvegardé le module à importer")
import pml_utils

from pml_utils import show_failures

# prédiction sur la base de test
predictions=model.predict(x_test)
show_failures(predictions, y_test, x_test, trueclass=9)
```

**Question : appliquer cette fonction pour toutes les classes CIFAR-10.**

Une autre manière d'appréhender la qualité du réseau et de vérifier la confusion que le réseau peut faire entre les classes. Pour cela nous construisons la matrice de confusion (confusion matrix). Dans cette matrice, chaque ligne correspond à une classe réelle et chaque colonne correspond à une classe estimée. Ainsi la cellule (ligne L, colonne C) contient le nombre d'éléments de la classe réelle L qui ont été estimés comme appartenant à la classe C. Cette matrice permet de voir ppour quelles classes le réseau est le plus ou le moins performant.
Voici comment établir cette matrice :

```
from sklearn.metrics import confusion_matrix
from numpy import argmax

print('Confusion matrix (rows: true classes; columns: predicted classes):'); print()
cm=confusion_matrix(y_test, argmax(predictions, axis=1), labels=list(range(10)))
print(cm); print()

print('Classification accuracy for each class:'); print()
for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)): print("%d: %.4f" % (i,j))

```

La diagonale regroupe les valeurs de précision pour chaque classe.

**Question : quelles classes semblent êtres les plus difficiles à prédire ?**









