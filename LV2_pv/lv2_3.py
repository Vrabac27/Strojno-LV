import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('tiger.png')

if img.max() <= 1.0:
    img = img * 255
img = img.astype(np.uint8)

img_bright = img.astype(np.int32) + 50
img_bright = np.clip(img_bright, 0, 255).astype(np.uint8)

img_rotated = np.rot90(img, k=-1)

img_mirrored = np.fliplr(img)

img_low_res = img[::10, ::10]

img_cropped_black = img.copy()
visina, sirina = img.shape[0], img.shape[1]
jedna_cetvrtina = sirina // 4
druga_cetvrtina = sirina // 2

img_cropped_black[:, :jedna_cetvrtina] = 0
img_cropped_black[:, druga_cetvrtina:] = 0

plt.figure()
plt.imshow(img_mirrored, cmap='gray', vmin=0, vmax=255)
plt.title("Zrcaljena slika")
plt.show()