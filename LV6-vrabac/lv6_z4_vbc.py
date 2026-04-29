import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans

imageNew = mpimg.imread('example_grayscale.png')

#Slika 2D matrica (visina, širina), kmeans treba 2D (broj_piksela, 1)
X = imageNew.reshape(-1, 1)

#kmeans (npr. 10 klastera)
n_colors = 10
kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
kmeans.fit(X)

#zamjena originalnih vrijednosti
compressed_image = kmeans.cluster_centers_[kmeans.labels_]
compressed_image = compressed_image.reshape(imageNew.shape)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(imageNew, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Kvantizirana slika ({n_colors} boja)")
plt.imshow(compressed_image, cmap='gray')
plt.show()