import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import datetime
import os

img_size = (48, 48)
batch_size = 64

print("Ucitavanje trening i validacijskog skupa...")
train_ds = image_dataset_from_directory(
    directory='gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    subset="training",
    seed=123,
    validation_split=0.2,
    image_size=img_size
)

validation_ds = image_dataset_from_directory(
    directory='gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    subset="validation",
    seed=123,
    validation_split=0.2,
    image_size=img_size
)

print("Ucitavanje testnog skupa...")
test_ds = image_dataset_from_directory(
    directory='gtsrb/Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False
)

model = models.Sequential()
model.add(layers.Rescaling(1./255, input_shape=(48, 48, 3)))

filter_sizes = [32, 64, 128]

for x in filter_sizes:
    model.add(layers.Conv2D(filters=x, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=x, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Dropout(rate=0.2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(43, activation='softmax'))

model.summary()
print(f"Ukupan broj parametara: {model.count_params():,}")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_callback = ModelCheckpoint(
    filepath='najbolji_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

print("\nPokretanje treninga...")
print(f"TensorBoard: tensorboard --logdir={log_dir}")
print("")

epochs = 5

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[checkpoint_callback, tensorboard_callback],
    verbose=1
)

# Graf tocnosti
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening tocnost')
plt.plot(history.history['val_accuracy'], label='Validacijska tocnost')
plt.title('Tocnost modela po epohama')
plt.xlabel('Epoha')
plt.ylabel('Tocnost')
plt.legend()
plt.grid(True)


# Graf gubitka
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening gubitak')
plt.plot(history.history['val_loss'], label='Validacijski gubitak')
plt.title('Gubitak modela po epohama')
plt.xlabel('Epoha')
plt.ylabel('Gubitak')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('trening_grafovi.png', dpi=150)
plt.show()


print("\nEvaluacija na testnom skupu...")
best_model = tf.keras.models.load_model('najbolji_model.keras')

test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"\nTocnost klasifikacije na testnom skupu: {test_acc * 100:.2f}%")
print(f"Gubitak na testnom skupu: {test_loss:.4f}")


print("\nGeneriranje predikcija za matricu zabune...")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = best_model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)


#izracun matrice zabune...")
cm = confusion_matrix(y_true, y_pred)
print("\nMatrica zabune:")
print(cm)

# graficki prikaz matrice zabune
plt.figure(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, include_values=False)
plt.title(f"Matrica zabune - ukupna tocnost: {test_acc*100:.2f}%")
plt.tight_layout()
plt.savefig('matrica_zabune.png', dpi=150)
plt.show()


print("\nIzvjestaj klasifikacije po klasama:")
print(classification_report(y_true, y_pred, target_names=[f'Klasa {i}' for i in range(43)]))

print("\n" + "=" * 50)
print("ZAVRSNI REZULTATI")
print("=" * 50)
print(f"Najbolji model: najbolji_model.keras")
print(f"TensorBoard logovi: {log_dir}")
print(f"Pokreni TensorBoard: tensorboard --logdir={log_dir}")
print(f"Tocnost na testnom skupu: {test_acc * 100:.2f}%")
print(f"Ukupno parametara: {best_model.count_params():,}")
print("=" * 50)