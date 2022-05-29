import sys

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data_fromlocalpath


def run_rna():
    # Tomo los datasets generados previamente
    (train_images_full, train_labels_full), (test_images, test_labels) = load_data_fromlocalpath('my-datasets')

    # Separo las imágenes de entrenamiento en entrenamiento/validación
    valid_images, train_images = train_images_full[:50], train_images_full[50:]
    valid_labels, train_labels = train_labels_full[:50], train_labels_full[50:]

    # Defino los labels de las clases (solo aplica para el ploteo de prueba)
    class_names = ['vacio', 'pdf417', 'otros']

    # Imprimir 25 primeras imagenes mediante plot
    plt.figure(figsize=(10, 10))
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[test_labels[i]])
    plt.show()

    valid_images, train_images, test_images = valid_images / 255., train_images / 255., test_images / 255.

    # Se define el modelo de la RNA con sus layers y neuronas
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                     keras.layers.Dense(300, activation='relu'),
                                     keras.layers.Dense(100, activation='relu'),
                                     keras.layers.Dense(100, activation='relu'),
                                     keras.layers.Dense(100, activation='relu'),
                                     keras.layers.Dense(10, activation='softmax')])

    # Se muestra cómo quedó definido el modelo
    model.summary()

    # Se compila el modelo
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # Se define el dataset de entrenamiento junto con sus labels. Se agrega también el dataset de validación y se
    # definen 10 épocas (iteraciones)
    history = model.fit(train_images,
                        train_labels,
                        epochs=10,
                        validation_data=(valid_images, valid_labels))

    # Se muestran los resultados de las iteraciones, en forma de texto y de gráfico cartesiano
    pd.DataFrame(history.history).plot(figsize=(16, 10))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    # Con lo anterior, se tiene al modelo entrenado. Se procede a ejecutar la evaluación con el dataset de prueba
    print('Evaluación final: ')
    model.evaluate(test_images, test_labels)


if __name__ == "__main__":
    run_rna()
else:
    print("Este script debe ejecutarse como como programa principal, no como módulo importado")
    sys.exit(1)
