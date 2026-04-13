import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


print("Prije skaliranja, X_train:")
print(X_train[0])
print("\nNakon skaliranja, X_train_s):")
print(X_train_s[0])
print(f"\nSrednja vrijednost X_train_s: {X_train_s.mean(axis=0)}")
print(f"Standardna devijacija X_train_s: {X_train_s.std(axis=0)}")

# Kreiraj i istreniraj KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_s, y_train)
# Predikcija modela
y_pred = knn.predict(X_test_s)
print("Predikcije za y_pred:")
print(y_pred)

# Izracunaj matricu zabune i prikazi ju
print("Matrica zabune:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Izracunaj preciznost
precision = precision_score(y_test, y_pred)
print("Preciznost: " + str(precision))

# Izracunaj odziv
recall = recall_score(y_test, y_pred)
print("Odziv: " + str(recall))

# Izracunaj tocnost
accuracy = accuracy_score(y_test, y_pred)
print("Tocnost: " + str(accuracy))

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train_s, y_train)

# Predikcija modela
y_pred = dt.predict(X_test_s)

# Vizualizacija stabla odlucivanja
plt.figure(figsize=(10, 6))
plot_tree(dt, feature_names=['S3_Temp', 'S5_CO2'], class_names=['Slobodna', 'Zauzeta'], filled=True)
plt.show()
