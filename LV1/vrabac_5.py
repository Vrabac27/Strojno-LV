file = open("C:/Users/student/Desktop/Vrabac/LV1/song.txt")

rijeci = {}

for line in file:
    words = line.split()
    for word in words:
        if word in rijeci:
            rijeci[word] += 1
        else:
            rijeci[word] = 1
file.close()

rijeci_jednom = []

for word, count in rijeci.items():
    if count == 1:  
        rijeci_jednom.append(word)
        
print(rijeci)
print("Broj riječi koje se pojavljuju samo jednom:", len(rijeci_jednom))
print("Te riječi su:", rijeci_jednom)