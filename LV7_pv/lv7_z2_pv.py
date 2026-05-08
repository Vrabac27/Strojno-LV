import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import keras
from keras import layers
from keras import ops


# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Labela: {y_train[i]}")
    plt.axis('off')
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()
# Define Sequential model with 3 layers
model = keras.Sequential()
model.add(layers.Dense(units=100, activation='relu', input_shape=(784,)))
model.add(layers.Dense(units=50, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# TODO: provedi treniranje mreze pomocu .fit()
model.fit(x_train_s, y_train_s, epochs=10, batch_size=32)


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
score_train = model.evaluate(x_train_s, y_train_s, verbose=0)
score_test = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Tocnost na skupu za ucenje: {score_train[1]:.4f}")
print(f"Tocnost na skupu za testiranje: {score_test[1]:.4f}")


# TODO: Prikazite matricu zabune na skupu podataka za testiranje
y_test_pred = model.predict(x_test_s)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

cm = confusion_matrix(y_test, y_test_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Matrica zabune")
plt.show()

# --- MATRICA ZABUNE ZA SKUP ZA UČENJE ---
y_train_pred = model.predict(x_train_s)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

cm_train = confusion_matrix(y_train, y_train_pred_classes)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
disp_train.plot(cmap='Reds') # Crvena boja za trening
plt.title("Matrica zabune - Skup za UČENJE")
plt.show()

# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala
errors = np.where(y_test_pred_classes != y_test)[0]
plt.figure(figsize=(10, 4))
for i, idx in enumerate(errors[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"P: {y_test_pred_classes[idx]}, S: {y_test[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Pronalaženje indeksa svih pogrešnih klasifikacija
errors = np.where(y_test_pred_classes != y_test)[0]

random_errors = np.random.choice(errors, size=5, replace=False)

plt.figure(figsize=(12, 5))
for i, idx in enumerate(random_errors):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray') # Reshape natrag u 28x28 za prikaz
    
    # P = Predicted (Procijenjeno), S = Stvarno (Actual)
    plt.title(f"P: {y_test_pred_classes[idx]}\nS: {y_test[idx]}")
    plt.axis('off')

plt.suptitle("Nasumični primjeri pogrešnih klasifikacija", fontsize=14)
plt.tight_layout()
plt.show()