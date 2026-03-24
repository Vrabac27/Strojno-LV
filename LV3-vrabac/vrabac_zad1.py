import pandas as pd
import numpy as np

mtcars = pd.read_csv('C:/Users/student/Desktop/LV3-vrabac/mtcars.csv')

#sortiro po potrosnji od najvece do najmanje
cars_sorted = mtcars.sort_values(by=['mpg'], ascending=False)
print(cars_sorted['car'].head(5))  #ispiso prvih 5 auta jer imaju najvecu potrosnju

# 2. Koja tri automobila s 8 cilindara imaju najmanju potrošnju?
cyl_8 = mtcars[mtcars.cyl == 8].sort_values(by='mpg')
print(cyl_8['car'].head(3))

#srednja potrosnja automobila s 6 cilindara
print(mtcars[mtcars.cyl==6].mpg.mean())

#srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs
print(mtcars[(mtcars.cyl==4) & (mtcars.wt>=2.000) & (mtcars.wt<=2.200)].mpg.mean())

# * am - Transmission (0 = automatic, 1 = manual)
print(mtcars.am.value_counts())


#Koliko je automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga?
am1_hp100 = mtcars[(mtcars.am==0) & (mtcars.hp>100) & (mtcars.cyl==4)]
print(len(am1_hp100))

#Kolika je masa svakog automobila u kilogramima suma
masa_kg = mtcars['wt'].sum() * 1000
print(str(masa_kg) + " kg")

#masa svakog automobila u kilogramima
mtcars['masa_kg'] = mtcars['wt'] * 1000
print(mtcars[['masa_kg']])
