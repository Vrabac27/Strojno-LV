from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# TODO: strukturiraj konvolucijsku neuronsku mrezu



# TODO: definiraj karakteristike procesa ucenja pomocu .compile()



# TODO: definiraj callbacks



# TODO: provedi treniranje mreze pomocu .fit()



#TODO: Ucitaj najbolji model


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje



# TODO: Prikazite matricu zabune na skupu podataka za testiranje
