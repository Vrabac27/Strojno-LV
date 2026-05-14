import numpy as np
from tensorflow import keras
from PIL import Image

# 1. Učitavanje  izgrađene mreže
model = keras.models.load_model('best_model.keras')

# 2. Učitavanje 'test.png'
# 28x28 piksela, siva skala
img = Image.open('test.png').convert('L')  # Pretvori u grayscale
img = img.resize((28, 28))                # Promijeni veličinu na 28x28
img_array = np.array(img)

# invertirati boje:
img_array = 255 - img_array 

#dodavanje dimenzija za batch
img_array = img_array.astype("float32") / 255
img_array = np.expand_dims(img_array, axis=0)
img_array = np.expand_dims(img_array, axis=-1)

# 3. Klasifikacija
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)

print(f"Mreža je klasificirala sliku kao znamenku: {predicted_digit}")