from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# TODO: strukturiraj konvolucijsku neuronsku mrezu
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TODO: definiraj callbacks
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs',
                                update_freq = 100),
    keras.callbacks.ModelCheckpoint(filepath='best_model.keras',
                                    monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True)
]

# TODO: provedi treniranje mreze pomocu .fit()
model.fit(x_train_s,
          y_train_s,
          epochs = 1,
          batch_size = 64,
          callbacks = my_callbacks,
          validation_split = 0.1)

#TODO: Ucitaj najbolji model
model = keras.models.load_model('best_model.keras')

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)

print(f"\nTočnost na skupu za učenje: {train_acc:.4f}")
print(f"Točnost na skupu za testiranje: {test_acc:.4f}")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
y_test_pred = np.argmax(model.predict(x_test_s), axis=1)
cm_test = confusion_matrix(y_test, y_test_pred)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.arange(10))

fig, ax = plt.subplots(figsize=(10, 8))
disp_test.plot(ax=ax, cmap=plt.cm.Blues)

plt.title('Matrica zabune na testnom skupu')
plt.show()

# 2. Matrica za TRENING skup
y_train_pred = np.argmax(model.predict(x_train_s), axis=1)
cm_train = confusion_matrix(y_train, y_train_pred)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=np.arange(10))

# Prikazivanje oba grafa
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

disp_train.plot(ax=ax[0], cmap=plt.cm.Blues)
ax[0].set_title('Matrica zabune - Skup za učenje')

disp_test.plot(ax=ax[1], cmap=plt.cm.Greens)
ax[1].set_title('Matrica zabune - Testni skup')

plt.show()