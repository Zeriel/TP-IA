# Clasificación de imágenes con una RNA tipo backpropagation

## Elementos de prueba
Para la tarea, se dispone de dos tipos de imágenes: las que corresponden a un código de barras PDF417, y las que no. Mediante el conversor de imágenes a formato MNIST, se clasifican de la siguiente forma:
 - 0 -> PDF417
 - 1 -> Otros

Cada uno de estos conjuntos está dividido en dos subgrupos: imágenes de entrenamiento e imágenes de prueba

 - Las imágenes de entrenamiento se utilizan para calibrar los nodos de la red.
 - Las imágenes de prueba se utilizan para validar si los nodos clasifican correctamente las imágenes dadas.


Tensorflow -> pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py3-none-any.whl
Loading Dataset: https://github.com/zalandoresearch/fashion-mnist/issues/167