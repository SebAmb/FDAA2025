# Premier projet en Keras - Regression logistique

Nous nous plaçons dans le cadre d'une régression logistique donc dans le cas d'un **classifieur binaire**.

Les étapes que vous apprendrez dans ce TP sont les suivantes :
  
  1. Charger des données
  2. Définir le modèle Keras
  3. Compiler le modèle Keras
  4. Compatible avec le modèle Keras.
  5. Évaluer le modèle Keras
  6. Attachez le tout ensemble
  7. Faire des prédictions

Vous utiliserez l'environnement Colab Research de Google: https://colab.research.google.com

Vous devez vous y connecter.

La première étape consiste à définir les fonctions et les classes que vous aller utiliser.

Vous utiliserez la bibliothèque NumPy et Sklearn pour charger votre jeu de données et deux classes de la bibliothèque Keras pour définir votre modèle.

Les importations requises sont répertoriées ci-dessous

```
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import seaborn as sns

import pandas as pd
```

## Chargement des données

Commencer par charger la dataset :
```
iris = datasets.load_iris()
print(iris.target)
```
Vous pouvez trouver des informations concernant cette dataset ici : [https://scikit-learn.org/stable/datasets/index.html#iris-dataset
](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
**Question : analyser la structure des données. Combien d'échantillons ? Combien de paramètres ?**

Pour produire un modèle et le tester sur les données, il faut transformer l'objet iris en numpy array. Attention, cette base de données comporte 
trois classes : 0, 1, 2. Pour se placer dans le cadre d'une classification binaire, nous allons donc dans un premier temps extraire cette base en fusionnant les classes 0 et 1.

```
X = iris["data"][:, 0:4]
print(X)

# On créé une liste qui contient 1 lorsque la fleur est de type 2 et 0 sinon pour faire une classification
y = iris["target"]
print(y)

# Grâce à cette ligne, la classe 2 est étiquetée 1 et les classes 0 et 1 sont étiquetées 0
y = (iris["target"] == 2)
print(y)
```

## Création du modèle

Nous allons commencer par créer un modèle séquentiel Kéras. Les modèles dans Keras sont définis comme une séquence de couches.

Nous créons un modèle séquentiel et ajoutons des couches une par une jusqu'à ce que nous soyons satisfaits de notre architecture réseau.

La première chose à faire est de s’assurer que la couche d’entrée possède le nombre correct de paramètres d'entrées. Cela peut être spécifié lors de la création de la première couche avec l'argument __input_shape__ et en le définissant sur __(4,)__ pour présenter les 4 paramètres d'entrée sous forme de vecteur.

Dans ce premie exemple nous utilisons une structure de réseau entièrement connectée avec une seule couche de sortie composées d'un seul neurone qui prend en entrée le vecteur complet de 4 paramètres.

Les couches entièrement connectées sont définies à l’aide de la classe __Dense__. Vous pouvez spécifier le nombre de neurones ou de nœuds dans la couche comme premier argument et la fonction d'activation à l'aide de l'argument d'activation (ici une sigmoid).

Autrefois, les fonctions d'activation Sigmoïde et Tanh étaient préférées pour toutes les couches. De nos jours, de meilleures performances sont obtenues grâce à la fonction d'activation ReLU. L'utilisation d'un sigmoïde sur la couche de sortie garantit que la sortie de votre réseau est comprise entre 0 et 1 et qu'elle est facile à mapper soit à une probabilité de classe 1, soit à une classification stricte de l'une ou l'autre classe avec un seuil par défaut de 0,5.

```
model = Sequential()

model.add(Dense(1, input_shape=(4,),activation='sigmoid'))
model.summary()
```
ou encore en séparant la couche d'entrée
```
# define the keras model
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

```

Attention, l'appel de la méthode __Sequential__  initialise le modèle à zéro. Par conséquent, si par la suite vous ajouter d'autres couches au réseau sans appeler cette méthode, elles viendront s'ajouter
aux couches que le modèle comportait déjà.

## Compilation du modèle


Maintenant que le modèle est défini, vous pouvez le compiler.

Lors de la compilation, vous devez spécifier certaines propriétés supplémentaires requises lors de la formation du réseau. N'oubliez pas que former un réseau signifie trouver le meilleur ensemble de pondérations pour mapper les entrées aux sorties de votre ensemble de données.

Vous devez spécifier la __fonction de perte__ à utiliser pour évaluer un ensemble de pondérations, __l'optimiseur__ utilisé pour rechercher différentes pondérations pour le réseau et toutes les __métriques facultatives__ que vous souhaitez collecter et signaler pendant l'entraînement.

Dans ce cas, utilisez l'entropie croisée comme argument de perte. Cette perte concerne un problème de classification binaire et est définie dans Keras comme __binary_crossentropy__. Vous pouvez en savoir plus sur le choix des fonctions de perte en fonction de votre problème ici : https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

Nous définirons l'optimiseur comme l'algorithme efficace de descente de gradient stochastique Adam. Il s'agit d'une version populaire de la descente de gradient car elle s'ajuste automatiquement et donne de bons résultats dans un large éventail de problèmes. Pour en savoir plus sur la version Adam de la descente de gradient stochastique, consultez l'article : https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

Enfin, comme il s'agit d'un problème de classification, vous collecterez et rapporterez la précision de la classification définie via l'argument métriques.

```
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

```

## Entraînement du modèle

Vous avez défini votre modèle et l'avez compilé pour vous préparer à un calcul efficace. Nous allons lancer l'apprentissage sur vos données chargées en appelant la fonction fit() sur le modèle.

La formation se déroule sur plusieurs époques et chaque époque est divisée en batchs.

     Epoch : un passage dans toutes les lignes de l'ensemble de données d'entraînement
     Batch : un ou plusieurs échantillons pris en compte par le modèle au cours d'une époque avant la mise à jour des poids

Une epoch comprend un ou plusieurs batchs, en fonction de la taille du batch choisi, et le modèle est adapté à plusieurs epoch : https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

Le processus de formation s'exécutera pendant un nombre fixe d'epoch (et d'itérations) à travers l'ensemble de données que vous devez spécifier à l'aide de l'argument epochs. Vous devez également définir le nombre de lignes de l'ensemble de données prises en compte avant la mise à jour des pondérations du modèle au sein de chaque époque, appelé taille du batch, et défini à l'aide de l'argument batch_size. Ce problème s'exécutera sur 150 epochs et utilisera une taille de lot de 10.

```
model.fit(X, y, epochs=150, batch_size=10)
```

## Evaluation du modèle

Vous avez formé notre réseau de neurones sur l'ensemble de données et vous pouvez évaluer les performances du réseau sur le même ensemble de données.

Cela vous donnera seulement une idée de la qualité de la modélisation de l'ensemble de données (par exemple, la précision du train), mais aucune idée de l'efficacité de l'algorithme sur de nouvelles données. Cela a été fait par souci de simplicité, mais idéalement, vous pourriez séparer vos données en ensembles de données d'entraînement et de test pour l'entraînement et l'évaluation de votre modèle.

Vous pouvez évaluer votre modèle sur votre ensemble de données d'entraînement à l'aide de la fonction évaluer() et lui transmettre les mêmes entrées et sorties que celles utilisées pour entraîner le modèle.

Cela générera une prédiction pour chaque paire d'entrée et de sortie et collectera des scores, y compris la perte moyenne et toutes les mesures que vous avez configurées, telles que la précision.

La fonction évaluer() renverra une liste avec deux valeurs. Le premier sera la perte du modèle sur l’ensemble de données, et le second sera la précision du modèle sur l’ensemble de données. Vous souhaitez uniquement signaler l'exactitude, alors ignorez la valeur de la perte.

```
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
```

Les réseaux de neurones sont des algorithmes stochastiques, ce qui signifie que le même algorithme sur les mêmes données peut entraîner un modèle différent avec des compétences différentes à chaque fois que le code est exécuté : https://machinelearningmastery.com/randomness-in-machine-learning/

La variance des performances du modèle signifie que pour obtenir une approximation raisonnable des performances de votre modèle, vous devrez peut-être l'ajuster plusieurs fois et calculer la moyenne des scores de précision : https://machinelearningmastery.com/evaluate-skill-deep-learning-models/

**Question : exécuter l'apprentissage plusieurs fois et noter l'accuracy à chaque fois. Que constatez vous ?**

Par ailleurs, la sortie du réseau est une valeur entre 0 et 1. Par conséquent, nous pouvons décider que la règle de décision utilisera un seuil de 0.5 et que si le score est inférieur à 0.5 alors la prédiction doit être 0 et 1 sinon : 

```
predictions = (model.predict(X) > 0.5).astype(int)
```

**Question : modifier la structure du réseau en ajoutant une seule couche dense cachée de 2, 4, 8, 16 ou 32 neurones. Sur cette couche vous utiliserez la fonction d'activation sigmoid puis Relu. Lancer à nouveau l'apprentissage.**

**Question : modifier la structure du réseau en ajoutant deux couches dense cachées de respectivement 32 et 64 neurones. Sur cette couche vous utiliserez la fonction d'activation sigmoid puis Relu. Lancer à nouveau l'apprentissage.**

**Question : que constater vous sur la durée de l'apprentissage, sur la valeur de loss et sur l'accuracy en opérant toutes les combinaisons relatives aux questions précédentes?**

# Second projet - application du précédent projet à un autre dataset

Modifier l'ensemble du script précédent en l'appliquant aux données contenues dans le fichier dataset_diabets.csv.
Pour cela vous utiliserez la fonction ```dataset = loadtxt('dataset_diabets.csv', delimiter=',')```

Il s'agit d'un ensemble de données d'apprentissage automatique standard du référentiel UCI Machine Learning. Il décrit les données du dossier médical des Indiens Pima et indique s'ils ont présenté ou non un diabète dans les cinq ans en fonction de certains paramètres. L'objectif est de définir un modèle capable de prédire l'état d'un patient en fonction des valeurs de ces paramètres.

Les informations sur cette base sont dans le fichier dataset_diabets.txt de ce dépot qu'il faudra donc chargé sur votre drive.

Il vous faudra bien contrôler le nombre de paramètres de chaque échantillon pour définir la dimension de la couche d'entrée du réseau.
Commencer par définir un réseau sans couche cachée. Puis opérer de la même manière que précédemment en ajoutant plusiers couches cachées et en modifiant leur nombre de neurones.

**Question : identique au projet précédent ... que constatez-vous sur la durée, l'évolution de la fonction de loss et l'accuracy ?**



