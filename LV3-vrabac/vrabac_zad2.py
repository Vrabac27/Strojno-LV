import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mtcars = pd.read_csv('C:/Users/student/Desktop/LV3-vrabac/mtcars.csv')

# Pomoću barplot-a prikažite na istoj slici potrošnju automobila s 4, 6 i 8 cilindara.

mpg_cyl = mtcars.groupby('cyl')['mpg'].mean()

plt.bar(mpg_cyl.index, mpg_cyl.values)

plt.title('Prosječna potrošnja automobila po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Potrošnja / mpg)')

plt.show()

#Pomoću boxplot-a prikažite na istoj slici distribuciju težine automobila s 4, 6 i 8 cilindara.

plt.title('Distribucija težine automobila po broju cilindara')
plt.ylabel('Težina (1000 lbs)')

plt.boxplot([mtcars[mtcars['cyl'] == 4]['wt'], #prvi box
             mtcars[mtcars['cyl'] == 6]['wt'], #drugi box
             mtcars[mtcars['cyl'] == 8]['wt']], #treći box
            labels=['4 cilindara', '6 cilindara', '8 cilindara']
           )
plt.show()