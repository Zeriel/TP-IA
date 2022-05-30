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

    # Normalizamos los datasets
    valid_images, train_images, test_images = valid_images / 255., train_images / 255., test_images / 255.

    # EXPLICACIÓN: Se recomienda trabajar con datasets normalizados (entre 0 y 1) para que
    # la solución converja en forma más rápida. Para esto, debería sumarse cada valor y dividirlo por la media, pero,
    # como sabemos que los datasets son imágenes de 28x28 pixeles, y cada pixel tiene una intensidad entre 1 y 255,
    # directamente dividimos cada datasets por 255 y ya queda normalizado.

    # Se define el modelo de la RNA con sus layers
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                     keras.layers.Dense(300, activation='relu'),
                                     keras.layers.Dense(100, activation='relu'),
                                     keras.layers.Dense(100, activation='relu'),
                                     keras.layers.Dense(100, activation='relu'),
                                     keras.layers.Dense(3, activation='softmax')])

    # Los métodos y parámetros significan:
    # - .Sequential = Modelo secuencial, es una secuencia de capas en una red
    # - Layer "Flatten" = La primer layer, es la de entrada y no es entrenable. Se encarga de transformar los datasets
    #   para que pueda trabajarlos la red.
    # - Layer "Dense" de 300 = Capa oculta intermedia del tipo "dense" con 300 neuronas. Emplea la función "relu"
    #   (rectificador / rectified linear activation function)
    # - Layers "Dense" de 100 = Idem a la anterior. Hay un rendimiento decreciente por cada nueva layer que se agrega,
    #   por lo que hay que tener cuidado y evaluar el agregado de nuevas capas a medida que se construye la red.
    # - Layer "Dense" de 3 = Capa final, con 3 neuronas porque son 3 clasificaciones (recordar la clasificación
    #   0 = vacío para que no rompa el código). Esta capa será la que prediga la clasificación de la imagen. Utiliza
    #   una función "softmax" (una función que convierte números en probabilidades que sumen 1)

    # Se muestra cómo quedó definida la estructura del modelo
    print('\nEstructura del modelo: \n')
    model.summary()

    # La estructura nos dice que el modelo posee 286.810 parámetros entrenables

    # Antes de entrenar el modelo, hay que compilarlo
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # En la compilación se define:
    # - La función de pérdida (loss function). Se utiliza "sparse categorical cross entropy", ya que poseemos un grupo
    # excluyente de labels que tratamos de predecir.
    # - La función de optimización que se usará contra la función de pérdida anterior. Se emplea SGD (Stochastic
    #   Gradient Descent), lo que asegurará que el modelo converja a una solución óptima, haciendo que Keras emplee
    #   el método de backpropagation.
    # - Por último, se agrega una métrica que nos diga qué tan bien se está desempeñando el modelo con las
    #   predicciones. Se elige como métrica la precisión (accuracy), que nos da un porcentaje de cuántas predicciones
    #   coinciden con la clase real.

    # Ahora si, se procede a entrenar el modelo
    print('\nSe inicia entrenamiento del modelo: \n')
    history = model.fit(train_images,
                        train_labels,
                        epochs=10,
                        validation_data=(valid_images, valid_labels))
    print('\nEntrenamiento finalizado\n')

    # Se define el dataset de entrenamiento junto con sus labels. Se agrega también el dataset de validación y se
    # definen 10 épocas (iteraciones). Una "época" es una pasada del dataset de entrenamiento por toda la red.
    # Keras permite definir adicionalmente un dataset de validación, el cual empleará al final de cada época para
    # comparar las predicciones de entrenamiento contra valores verdaderos para evaluar su desempeño

    # Una vez se encuentra entrenada la red, se puede mostrar cómo le fue en el entrenamiento, mostrando en cada
    # iteración cómo fueron variando los parámetros.
    print('\nResultados del entrenamiento (ver gráfico generado por panda): \n')
    pd.DataFrame(history.history).plot(figsize=(16, 10))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    # Los parámetros que muestra el gráfico son:
    # - La pérdida en cada iteración, y la pérdida respecto al dataset de validación.
    # - La precisión en cada iteración, y la precisión respecto al dataset de validación.

    # Como se puede ver, a medida que se itera la precisión converge hacia 1, mientras que la pérdida decrece. Esto
    # significa que el entrenamiento de la red es efectivo y puede clasificar correctamente las imágenes.

    # Finalmente, con el modelo entrenado, se procede a evaluar qué tan bien se desenvuelve con el dataset de prueba.
    print('\nEvaluación final: \n')
    model.evaluate(test_images, test_labels)

    # Hay que recordar que hasta ahora siempre se usó el dataset de entrenamiento (con su subdivisión de validación),
    # el dataset de prueba es un nuevo dataset que hasta ahora nunca fue visto ni usado por el modelo. Por lo tanto,
    # esta evaluación es una prueba real de clasificación de las imágenes basándose en el entrenamiento previo de la
    # red.


if __name__ == "__main__":
    run_rna()
else:
    print("Este script debe ejecutarse como como programa principal, no como módulo importado")
    sys.exit(1)
