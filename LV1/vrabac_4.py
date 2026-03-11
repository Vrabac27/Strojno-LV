name = input("Unesi ime datoteke: ")

file = open(name)

sum=0
counter=0

for line in file:
    if line.startswith("X-DSPAM-Confidence:"):
        number = float(line.split(":")[1])
        sum += number
        counter += 1
        
prosjek = sum / counter

print("Average X-DSPAM-Confidence:", prosjek)
        