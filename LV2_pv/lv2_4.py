import numpy as np
import matplotlib.pyplot as plt

def kreiraj_sahovsku_plocu(velicina_kvadrata, br_kvadrata_visina, br_kvadrata_sirina):
    crni = np.zeros((velicina_kvadrata, velicina_kvadrata))
    bijeli = np.ones((velicina_kvadrata, velicina_kvadrata)) * 255
    
    par1 = np.hstack((crni, bijeli))
    par2 = np.hstack((bijeli, crni))
    
    redovi = []

    for i in range(br_kvadrata_visina):
        linija_kvadrata = []
        for j in range(br_kvadrata_sirina):
            if (i + j) % 2 == 0:
                linija_kvadrata.append(crni)
            else:
                linija_kvadrata.append(bijeli)
        
        cijeli_red = np.hstack(linija_kvadrata)
        redovi.append(cijeli_red)

    konacna_slika = np.vstack(redovi)
    return konacna_slika

slika_ploce = kreiraj_sahovsku_plocu(50, 4, 5)

plt.figure()
plt.imshow(slika_ploce, cmap='gray', vmin=0, vmax=255)
plt.show()