import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 3.0, 3.0, 2.0, 1.0])
y = np.array([1.0, 1.0, 2.0, 2.0, 1.0])

plt.plot(x, y, 'o-b', linewidth=2)

plt.title("Primjer")
plt.xlabel("x os")
plt.ylabel("y os")

plt.xlim(0.0, 4.0)
plt.ylim(0.0, 4.0)

plt.show()