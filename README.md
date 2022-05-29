# Clasificación de imágenes con una RNA tipo *backpropagation*

Este repositorio contiene el código necesario para entrenar una red neuronal artifical del tipo *backpropagation* que logre identificar si una imagen dada es un código de barras PDF417.

## Elementos de prueba
Para la tarea, se dispone de dos tipos de imágenes: las que corresponden a un código de barras PDF417, y las que no. Mediante el conversor de imágenes a formato MNIST, se clasifican de la siguiente forma:
 - 1 -> PDF417
 - 2 -> Otros

Cada uno de estos conjuntos está dividido en dos subgrupos: imágenes de entrenamiento e imágenes de prueba

 - Las imágenes de entrenamiento se utilizan para calibrar los nodos de la red.
 - Las imágenes de prueba se utilizan para validar si los nodos clasifican correctamente las imágenes dadas.

## Utilización del conversor
El conversor (ubicado en ./conversor-ubyte) se toma de esta fuente: https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format/blob/master/convert-images-to-mnist-format.py.

Fueron necesarios unos fixes y retoques para que el conversor funcionase. Para empezar, la clase "0" no era tenida en cuenta, por lo que se definieron 3 clases de imágenes:
 - 0 -> Vacío, para que no rompa
 - 1 -> Imágenes correspondientes a PDF417
 - 2 -> Imágenes correspondientes a otra cosa

Para el caso "1", se emplearon 50 imágenes de códigos PDF417, divididas en 40 de entrenamiento y 10 de prueba.
Para el caso "2", se emplearon 100 imágenes de códigos QR, divididos en 60 de entrenamiento y 40 de prueba.

El conversor se compone de dos pasos: hacer un *resize* de las imágenes a 29x29 (tamaño estándar de MNIST) y luego comprimirlas en archivos *ubyte*. Dichos pasos están definidos en el README.md:
 - conversor-ubyte/conversor_app/README.md

Como el conversor solo opera con Python 2.7, se agregó un *Dockerfile* con la imagen Python:2.7 y la instalación de las dependencias del conversor, y un *docker-compose.yml* para levantar el contenedor. Los comandos de *resize* y *python* deben ejecutarse dentro del contenedor, y mediante la definición de un volumen generarán los *datasets* en su computadora en:
 - conversor-ubyte/conversor_app


## Carga de datasets generados por el conversor
Los *datasets* que genera el conversor no pueden agregarse nativamente, se tuvo que emplear una función provista por un tercero de GitHub:
 - Loading Dataset: https://github.com/zalandoresearch/fashion-mnist/issues/167

Dicha función se define en el archivo *utils.py*

*IMPORTANTE: Para que el proyecto principal tome los datasets, deben moverse de "conversor-ubyte/conversor_app" a "my-datasets"*

## Uso del proyecto
Se debe ejecutar el archivo *main.py* para iniciar el proceso. El *script* tomará los datasets del directorio "my-datasets", los cargará en variables, y comenzará los procesos de calibración de la red neuronal mediante los *datasets* de entrenamiento, y la validación del entrenamiento mediante los *datasets* de prueba.

El *dataset* de entrenamiento se subdivide en entrenamiento y validación, para que la red pueda aprender mejor mediante la validación de datos contra los que está usando para entrenar.