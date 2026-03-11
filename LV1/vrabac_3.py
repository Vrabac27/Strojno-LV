numbers = []

while True: 
    num_input= input("Unesi brojeve ili 'Done' za prekid: ")
    
    if num_input == "Done":
        break
    
    try:
        number = float(num_input)
        numbers.append(number)
    except:
        print("Greška! Nije unesen broj.")

if len(numbers) > 0: 
    print("Uneseni brojevi:", numbers)
    print("Broj unesenih brojeva:", len(numbers))
    print("Srednja vrijednost:", sum(numbers)/len(numbers))
    print("Minimalna vrijednost:", min(numbers))
    print("Maksimalna vrijednost:", max(numbers))
    numbers.sort()
    print("Sortirana lista:", numbers)
    
else:
    print("Nije unesen nijedan broj.")