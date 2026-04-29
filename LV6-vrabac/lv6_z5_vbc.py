import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans

try:
    image = mpimg.imread('example.png')
except FileNotFoundError:
    print("Greška: Datoteka 'example.png' nije pronađena.")
    # Za demonstraciju kreiramo nasumičnu sliku ako datoteka ne postoji
    image = np.random.rand(300, 300, 3)

h, w, c = image.shape
X = image.reshape(-1, c)

n_colors = 8 
kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
kmeans.fit(X)

quantized_colors = kmeans.cluster_centers_[kmeans.labels_]

quantized_image = quantized_colors.reshape(h, w, c)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Kvantizirana slika ({n_colors} boja)")
plt.imshow(quantized_image)
plt.axis('off')

plt.tight_layout()
plt.show()