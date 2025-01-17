# Premier projet en Keras - Regression logistique - Suppléments

Dans ce sujet, nous allons étudier quelques spécificités de Keras et notamment les points suivants :

1. Spécifier un sous-ensemble de validation
2. Afficher les courbes d'apprentissage
3. Tester différents algorithmes d'apprentissage
4. Enregistrer/charger les poids et l'architrcture du réseau
5. Optimisation des données d'apprentissage

Pour cela vous reprendrez les lignes de code que vous avez développés dans le sujet précédent
pour faire un script complet d'un classifieur MLP binaire dans lequel vous assurerez les étapes suivantes :

1. Chargement de la dataset IRIS avec sa modification sur deux classes
2. Définition et compilation de l'architecture du réseau MLP
3. Lancement de l'entrainement
4. Evaluation du modèle obtenu

## Spécification de la base de validation

Durant l'apprentissage, les données de la base sont présentées au réseau un nombre __n__ de fois (__n__ epochs). La base est découpée en plusieurs
batchs, chaque batch contenant un nombre __m__ d'échantillons. Par conséquent, pour une base d'apprentissage de __L__ échantillons
et si toute la base est utilisée pour apprendre le modèle, à chaque epoch __L/m__ batchs sont présentés au réseau. Un batch présenté correspond à une itération. 

Lorsque vous lancez la méthode ```model.fit(X, y, epochs=..., batch_size=...)``` de votre modèle alors vous obtenez la sortie suivante :

```
Epoch 1/200
15/15 [==============================] - 1s 4ms/step - loss: 0.6153 - accuracy: 0.7133
Epoch 2/200
15/15 [==============================] - 0s 3ms/step - loss: 0.5432 - accuracy: 0.8533
```

**RAPPEL** Dans cette configuration, à chaque itération, sont calculées la valeur de loss et la valeur d'accuracy, toutes les deux calculées à partir de la base d'apprentissage.
Les valeurs de loss sont calculées en utilisant la fonction loss précisée lors de l'appel de la méthode compile() du modèle. Les valeurs d'accuracy sont calculées en utilisant
la métrique précisée lors de l'appel de la méthode compile().

Lors de l'apprentissage il est requis de préciser une base de données de validation. Voici deux manières de le faire.

_Méthodes 1_ : préciser le paramètre __validation_split__ entre 0 et 1. Une valeur de 0.2 définit une base de validation composée de 20% des échantillons de la base d'apprentissage.

```
history=model.fit(X, y, epochs=500, batch_size=10, shuffle=False, validation_split=0.0)
```
La sortie précise alors les valeus de loss et d'accuracy pour la base d'apprentissage restante (i.e. 80%) et la base de validation. Rappelons que seule les 80% servent à
mettre à jour directement ls poids du réseau. La base de validation est utilisée pour qualifier le pouvoir de généralisation du réseau. Cette qualification s'effectue 
à chaque epoch grâce aux 4 valeurs loss, accuracy, val_loss et val_accuracy. La sortie obtenue est la suivante :
```

Epoch 5/500
12/12 [==============================] - 0s 6ms/step - loss: 0.0325 - accuracy: 0.9917 - val_loss: 0.3693 - val_accuracy: 0.7667
Epoch 6/500
12/12 [==============================] - 0s 5ms/step - loss: 0.0327 - accuracy: 0.9917 - val_loss: 0.3862 - val_accuracy: 0.7667
```

_Méthode 2_ : il suffit de créer un sous-ensemble de données de validation à partir de la base d'apprentissage et de le passer à la méthode fit() du modèle.
Pour cela nous allons utiliser la librairie sklearn et la fonction train_test_split() au travers des lignes de code suivantes :
```
# Création des sous ensemble de validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)
```
Dans cet exemple, nous créons les deux sous-ensembles X_train et X_val et leurs labels respectifs y_train et y_val. Les échantillons de la base de validation X_val représentent 20% de la base d'apprentissage initiale X. L'extraction est réalisée de manière aléatoire en précisant le seed du générateur de nombres pseudo-aléatoires (random_state=2).En changeant cette valeur, vous obtiendrez des sous-ensembles différents.

## Affichage des courbes d'apprentissage

Voici quelques lignes pour tracer la valeur de loss, accuracy, val_loss et val_accuracy obtenues à chaque époch.
Pour cela, nous devons récupérer ce qui est retourné par la méthode fit().
```
...
history=mpodel.fit(...)
...
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.plot(history.history['val_loss'], label = 'val loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
```

**Question : Afficher les courbes pour l'architectures MLP à deux couches cachées dont vous modifierez le nombre de neurones selon les configirations suivantes : (64,32) (32,16)(16,8)(8,4).
A partir de ces 4 courbes, il est alors possible de constater ou non l'apparition ou non d'un surapprenissage (overfitting). Un surapprentissage est le résultat d'une inadéquation entre le nombre de poids 
du réseau et le nombre d'instance de la base d'apprentissage. Une fois vos 4 courbes disponibles, appelez-moi pour une explication.**

**Question : tracer ces courbes pour le MLP à deux couches cachées (32,16) et une taille de batch de 1. Que constatez vous ?**

## Tester plusieurs algorithmes d'apprentissage

Test votre réseau en fonction de l'algo d'optimisation utilisé   : SGD, RMSprop, Adam, AdamW, Adagrad

Faite également varier le __learning_rate__ qui qui joue sur la rapidité de la descente de gradient. C'est un paramètre qui peut poser de gros problème : nous allons être face à une oscillation des performance ou tomber dans un minimum local. Par défaut la valeur du learning_rate est 0.001. Voici un exemple dans le cas d'Adam :

```
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt)
```

**Question : analyser l'accuracy pour plusieurs valeurs de learning_rate et pour plusieurs algo d'apprentissage**


## Enregistrement/chargement des poids du réseau appris

Une fois la phase d'entraînement terminée, il est possible d'enregistrer les poids du réseau et son architecture grâce à ```model.save("regressionML.keras")```. Cette fonction sauvegarde l'architecture du réseau et les poids du réseau obtenus à la toute dernière itération dans un fichier compressé. Il est possible de ne sauvegarder que les poids ```model.save_weights("regressionMLP.weights.h5")``` dans un fichier au format .h5.

Une fois les sauvegardes effectuées, il est alors possible de charger à nouveau les réseaux pour réaliser d'autres inférences, sur d'autres plateformes et d'autres données. Le fichier .keras permet, en le chargeant, de décrire à la fois l'architecture et les poids. le fichier .h5  permet de ne charger que les poids et nécessite donc de décrire préalablement et manuellement l'architecture du réseau comme vouz l'avez fait jusque là. 

```
# Charger un modèle
restored_model = keras.models.load_model("regressionMLP.keras")

ou

# Charger les poids d'un modèle
# Description du réseau
model = Sequential()
model.add(Dense(128, input_shape=(4,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.load_weights("regressionMLP.weights.h5")
```
Ainsi, il est possible d'inférer le réseau chargé sur de nouvelles données de la manière suivante :
```
# Prédiction sur le sous-ensemble d'apprentissage
res=model.predict(X_train)
res2=restored_model.predict(X_train)

# Prédiction sur deux nouvelles instances X_1 et X_2
X_1=[5.8,2.6,4.0,1.2]
X_2=[6.3,3.3,6.0,2.5]
res1=model.predict([X_1])
res2=model.predict([X_2])
print('classe {0} and classe {1}'.format(round(res1[0,0]),round(res2[0,0])))
```

## Sauvegarde des poids durant l'apprentissage

Il est possible de sauvegarder les poids du réseau durant l'apprentissage. Pour cela, nous devons créer une fonction callback qui sera appelée par la fonction .fit()
Il est possible de sauvegarder les poids estimés à la fin de chaque epoch (https://keras.io/api/callbacks/model_checkpoint/)
Toutefois, pour éconoimser de l'espace mémoire (tout particulièrement pour les gros réseaux) il est possible de sauvegarder les poids du réseau sous conditions. Par exemple, nous
choisirons de sauvegarder uniquement si la valeur de val_loss est plus faible que la valeur obtenue à l'epoch précédente.

```
# construction de la fonction callback qui sauvegarde les poids
# seulement si le modèle est meilleur sur la base de val_loss
checkpoint = keras.callbacks.ModelCheckpoint(filepath="./weights/weights-{epoch:03d}-{val_loss:.4f}.weights.hdf5", monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True, verbose=1)
callbacks = [checkpoint]

# appel de la nouvelle fonction de fit()
history=model.fit(X, y, epochs=200, batch_size=10, shuffle=False, validation_split=0.2, callbacks=callbacks, verbose=2)
```

Il est possible de changer la valeur qui est monitorée pour créer la condition de sauvegarde : ```fit(...,monitor=val_accuracy,mode="max",verbose=1)```

Attention dans la version Keras du moment, pour n'enregistrer que les poids, il est requis de spécifier le paramètre ```save_weights_only=True``` et le nom du fichier de poids doit se terminer par ```weights.hdf5``` dans la définition de la callback.

**Question : vérifier la bonne sauvegarde des poids dans le dossier que vous avez préciseé (ici ./weights/weights-032-0.4321.hdf5)**

**Question : une fois l'entraînement terminé, charger le dernier fichier de poids sauvegardé avec la callback et inférer ce réseau sur la base X_train afin d'en calculer la précision
comme vous l'avez fait dans le sujet précédent**

## Standardisation des données

L'échelle et la dynamique des données d'apprentissage seront probablement très variables. Par conséquent une étape de standardisation des données est requise afin de réduire l'effet de 
cette variabilité sur l'entraînement. Nous la réalisons avec la bibliothèque sklearn et la fonction StandardScaler(). La standardisation permet de recentrer les données et de ramener la variance des données
à 1.

```
from sklearn.preprocessing import StandardScaler

# en reprenant les X_train, X_val, y_train, y_val précédents
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)

# entraînement sur le dataset standardisé
history=model.fit(X_train_scaled, y_train, epochs=350, batch_size=10, shuffle=False, validation_data=(X_val_scaled,y_val), callbacks=callbacks)
```

Le modèle ainsi appris n'est alors inférable que sur des nouvelles données qui ont été standardisées également avec les mêmes transformations (moyenne et variance). En reprenant
les lignes de code précédentes et en considérant que l'objet scaler est encore disponible :

```
X_1=scaler.transform(np.array([[5.8,2.6,4.0,1.2]]))
X_2=scaler.transform(np.array([[6.3,3.3,6.0,2.5]]))
res1=model.predict([X_1])
res2=model.predict([X_2])
print('classe {0} and classe {1}'.format(round(res1[0,0]),round(res2[0,0])))
```

Dans la plupart des cas, vous aurez entrainé le réseau sur une base d'apprentissage que vous aurez standardisée et vous aurez sauvegardé la valeur de la moyenne et de la variance de la transformation qui vous aura servi à standardiser les données d'apprentissage. Ces deux paramètres vous servirons alors à transformer les nouvelles données sur lesquelles vous souhaiteriez inférer le réseau :

```
std  = np.sqrt(scaler.var_)
mean = scaler.mean_
X=np.array([[6.3,3.3,6.0,2.5]])
X_1_scaled = (X_1 - mean)) / std

res1=model.predict([X_1_scaled])
print('classe {0} and classe {1}'.format(round(res1[0,0]),round(res2[0,0])))

```

## Passage au cas multi classes

Dans ce dernier exercice, nous allons repasser la dataset Iris à 3 classes et entrainer une nouvelle architecture MLP.

Pour cela vous allez devoir faire les modifications suivante.
1. La dernière couche dense du modèle doit être composée de 3 neuones : chaque neurone prendra la valeur 1 pour la classe correspondance (100 - 010 - 001).
2. Changer le format des labels y : chaque label sera codé avec un tableau de 3 valeurs et la nouvelle variable des labels sera appelée ```y_cat = keras.utils.to_categorical(y, 3)```
3. Changer la fonction de loss du model.compile(...) : loss="categorical_crossentropy"
4. Après l'entraînement sur l'ensemble des échantillons X (avec label y_cat), lancer une évaluation du modèle sur ce même ensemble et calculer sa précision.

Pour terminer, vous afficherez pour chaque échantillon, la prédiction du modèle et le label attendu : 

```
# Calculer la prédiction de la classe pour tous les échantillons
predictions = (model.predict(X) > 0.5).astype(int)

# Afficher les prédictions et les labels
for i in range(150):
  if (np.abs((predictions[i]-y_cat[i])).sum())!=0:
    print(X[i],"-",predictions[i],"-", y_cat[i])
```

## Passage à la dataset MNIST

Cette dataset contient 10 classes, chacune correspondant à un chiffre manuscrit.
Appliquer le réseau MLP multi-classes précédent à cette dataset.

Pour charger la dataset : 
```
import tensorflow.keras.datasets.mnist as mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```
Ici les images sont normalisées en divisant par 255, de manière similaire à la standardisation que nous avons appliquée aux données Iris.

Attention, vous avez ici une base d'apprentissage (X_train, y_train) et une base de test (X_test,y_test).
Seule la base X_train devra être utilisée pour l'entrainement. Donc vous devrez en extraire une base de validation (X_val, y_val)
que vous poasserez à la fonction .fit().

Notons que les images sont des données en 2D de dimension 28x28.
Or l'entrée de notre MLP doit être 1D. Par conséquent, nous allons "applatir" les images de la base d'apprentissage. Pour cela nous
allons ajouter une couche __flatten__ de la manière suivante :
```
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

```
**Question : Afficher les courbes d'apprentissage sur les configurations du MLP suivantes (256,128) (128,64) (64,32) (32,16)**

**Question : A l'issue de l'entrainement et en analysant les courbes d'apprentissage, vous choisirez ce qui vous semble être le meilleur réseau
(composé des meilleurs poids) et vous l'évaluerez sur la base de test (X_test, y_test) en calculant son accuracy.**

**Question : Finalement vous produirez plusieurs images de chiffre de votre plus belle écriture (ou non) sur lesquelles vous inférerez
le réseau que vous aurez retenu. Ainsi, vous vérifierez qu'il prédit correctement les chiffres que vous aurez écrits. Attention, les chiffres devront être
blancs sur fond noir et vous devrez normaliser vos images (/255).**

Pour cette dernière question, il faut veiller à bien ecapsuler l'image sur laquelle vous souhaitez inférer le réseau. Ainsi vous pourrez commencer par inférer sur 
une image de la base de test de la manière suivante :
```
from keras.preprocessing.image import img_to_array
from numpy import argmax

# l'image est transformée en tableau
img=img_to_array(x_test[10])

# le tableau est transformée en tableau de tableaux (un tableau d'images)
img=img.reshape(1,28,28)

# prédiction du modèle sur img
res=model.predict(img)
# affichage de la prédiction 
print(res)

# le nombre inscrit correspond à la classe pour laquelle la valeur est maximale : utlisation de la fonction argmax 
digit = argmax(res)

# affichage de la réponse et de la bonne réponse
print("la prédiction est ",digit)
print("le label de l'image est ",y_test[10])

```




