import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)

mpg = data[:, 0]
cyl = data[:, 1]
hp = data[:, 3]
wt = data[:, 5]

plt.figure()
plt.scatter(hp, mpg, s=wt * 25, color='blue', alpha=0.7)
plt.xlabel("Konjska snaga (hp)")
plt.ylabel("Potrošnja (mpg)")
plt.title("Ovisnost potrošnje o konjskoj snazi i težini vozila")
plt.show()

min_mpg = np.min(mpg)
max_mpg = np.max(mpg)
mean_mpg = np.mean(mpg)

print("Sva vozila:")
print(f"Minimalni mpg: {min_mpg}")
print(f"Maksimalni mpg: {max_mpg}")
print(f"Srednji mpg: {mean_mpg}\n")

maska_6_cilindara = (cyl == 6)
mpg_6_cilindara = mpg[maska_6_cilindara]

min_mpg_6 = np.min(mpg_6_cilindara)
max_mpg_6 = np.max(mpg_6_cilindara)
mean_mpg_6 = np.mean(mpg_6_cilindara)

print("Vozila sa 6 cilindara:")
print(f"Minimalni mpg: {min_mpg_6}")
print(f"Maksimalni mpg: {max_mpg_6}")
print(f"Srednji mpg: {mean_mpg_6}")