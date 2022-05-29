from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data_fromlocalpath

# Tomo los datasets generados previamente
(train_images_full, train_labels_full), (test_images, test_labels) = load_data_fromlocalpath('my-datasets')

print(train_images_full.size)
print(train_labels_full.size)
print(test_images.size)
print(test_labels.size)

valid_images, train_images = train_images_full[:50], train_images_full[50:]
valid_labels, train_labels = train_labels_full[:50], train_labels_full[50:]

class_names = ['vacio', 'pdf417', 'otros']

# Imprimir 25 primeras imagenes
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

model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.Dense(300, activation='relu'),
                                 keras.layers.Dense(100, activation='relu'),
                                 keras.layers.Dense(100, activation='relu'),
                                 keras.layers.Dense(100, activation='relu'),
                                 keras.layers.Dense(10, activation='softmax')])

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(train_images,
                    train_labels,
                    epochs=10,
                    validation_data=(valid_images, valid_labels))

pd.DataFrame(history.history).plot(figsize=(16, 10))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

print('Evaluación final: ')
model.evaluate(test_images, test_labels)
