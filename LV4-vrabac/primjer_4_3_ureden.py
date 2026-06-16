import pandas as pd

# 1. Učitaj podatke
df = pd.read_csv('cars_processed.csv')

print("ODGOVORI NA PITANJA (ZADATAK 4):")

# 3. Najveća i najmanja cijena
indeks_najskuplji = df['selling_price'].idxmax()
indeks_najjeftiniji = df['selling_price'].idxmin()
print("3. Najskuplji auto je:", df.loc[indeks_najskuplji, 'name'])
print("   Najjeftiniji auto je:", df.loc[indeks_najjeftiniji, 'name'])

# 4. Automobili proizvedeni 2012.
auti_2012 = df[df['year'] == 2012]
print("4. Broj automobila iz 2012. godine:", len(auti_2012))

# 5. Najviše i najmanje kilometara
indeks_najvise_km = df['km_driven'].idxmax()
indeks_najmanje_km = df['km_driven'].idxmin()
print("5. Najviše kilometara prošao je:", df.loc[indeks_najvise_km, 'name'])
print("   Najmanje kilometara prošao je:", df.loc[indeks_najmanje_km, 'name'])

# 6. Najčešći broj sjedala
print("6. Najčešći broj sjedala je:", int(df['seats'].mode()[0]))

# 7. Prosječna kilometraža (Dizel vs Benzin)
dizelaši = df[df['fuel'] == 'Diesel']
benzinci = df[df['fuel'] == 'Petrol']
print(f"7. Prosjek kilometara za Dizel: {dizelaši['km_driven'].mean():.2f} km")
print(f"   Prosjek kilometara za Benzin: {benzinci['km_driven'].mean():.2f} km")