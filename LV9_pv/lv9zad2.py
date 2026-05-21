import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Ucitavanje modela
model = tf.keras.models.load_model('najbolji_model.keras')

# Imena znakova (redom od klase 0 do 42)
znakovi = [
    "Ogranicenje 20", "Ogranicenje 30", "Ogranicenje 50", "Ogranicenje 60",
    "Ogranicenje 70", "Ogranicenje 80", "Ogranicenje 100", "Ogranicenje 120",
    "Zabrana pretjecanja", "Zabrana pretjecanja kamiona", "Krizanje s pravom prolaza",
    "Glavna cesta", "Daj prednost", "Stop", "Zabrana prometa",
    "Zabrana ulaza", "Ogranicenje visine", "Opasnost", "Skretanje lijevo",
    "Skretanje desno", "Skretanje ravno", "Obilazak s desna", "Obilazak s lijeva",
    "Biciklisti", "Pjesaci", "Skola", "Prednost za pjesake", "Radovi na cesti",
    "Kraj ogranicenja 30", "Kraj ogranicenja 50", "Kraj ogranicenja 80",
    "Kraj zabrane pretjecanja", "Kraj zabrane pretjecanja kamiona", "Obavezno ravno",
    "Obavezno desno", "Obavezno lijevo", "Obavezno ravno ili desno",
    "Obavezno ravno ili lijevo", "Obavezno desno ili lijevo", "Kruzni tok",
    "Kraj glavne ceste", "Znak za zaustavljanje"
]

# Putanja do slike (PROMIJENI AKO TREBA)
test_image_path = 'gtsrb/Test/1/00001.png'

# Ucitaj i obradi sliku
img = Image.open(test_image_path)
if img.mode != 'RGB':
    img = img.convert('RGB')

img_resized = img.resize((48, 48))
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predikcija
predikcija = model.predict(img_array, verbose=0)
predvidjena_klasa = np.argmax(predikcija[0])
pouzdanost = np.max(predikcija[0]) * 100

# Prikazi sliku
plt.imshow(img)
plt.title(f"Predvidjeno: {znakovi[predvidjena_klasa]}")
plt.axis('off')
plt.show()

# Ispis rezultata
print("=" * 50)
print(f"Predvidjeni znak: {znakovi[predvidjena_klasa]}")
print(f"Pouzdanost: {pouzdanost:.1f}%")
print("=" * 50)