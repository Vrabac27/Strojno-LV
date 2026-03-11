def total_pay(hours, rate):
    total = hours*rate
    return total

hours = float(input("Unesi radne sate:"))
rate = float(input("e/h:"))

result = total_pay(hours, rate)

print ("Ukupna zarada: ", result, "e.")