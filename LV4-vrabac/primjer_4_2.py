import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy

x = np.linspace(1, 10, 50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

np.random.seed(12)
indeksi = np.random.permutation(len(x))
indeksi_train = indeksi[0:int(np.floor(0.7*len(x)))]
indeksi_test = indeksi[int(np.floor(0.7*len(x))):len(x)] 

degrees = [2, 6, 15]

MSEtrain = []
MSEtest = []

plt.figure(figsize=(10, 6))
plt.plot(x, y_true, 'k--', linewidth=2, label='Pozadinska funkcija (stvarno)')
plt.scatter(x[indeksi_train], y_measured[indeksi_train], color='blue', alpha=0.5, label='Podaci za učenje')

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(x)
    
    # Razdvajanje na train i test skup
    xtrain = x_poly[indeksi_train]
    ytrain = y_measured[indeksi_train]
    xtest = x_poly[indeksi_test]
    ytest = y_measured[indeksi_test]
  
    linearModel = lm.LinearRegression()
    linearModel.fit(xtrain, ytrain)

    ytrain_p = linearModel.predict(xtrain)
    ytest_p = linearModel.predict(xtest)

    MSEtrain.append(mean_squared_error(ytrain, ytrain_p))
    MSEtest.append(mean_squared_error(ytest, ytest_p))

    y_plot = linearModel.predict(x_poly)
    plt.plot(x, y_plot, label=f'Model (degree={deg})')

print("Stupnjevi polinoma: ", degrees)
print("Vektor MSEtrain:    ", [f"{mse:.4f}" for mse in MSEtrain])
print("Vektor MSEtest:     ", [f"{mse:.4f}" for mse in MSEtest])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Usporedba modela različitih stupnjeva s pozadinskom funkcijom')
plt.legend(loc='best')
plt.ylim(np.min(y_measured)-2, np.max(y_measured)+2) # Ograničenje y-osi zbog oscilacija deg=15
plt.show()