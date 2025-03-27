#8.1

dividend=int(input('Enter the dividend number:'))
divisor=int(input('Enter the divisor:'))
try:
    print(dividend/divisor)
except ZeroDivisionError:
    print("ERROR-CAN'T PRINT BY 0.")

#8.2

try:
    num=int(input("Enter a number: "))
except ValueError:
    print("Invalid input! Please enter a number.")

#8.3

dividend1=int(input('Enter the dividend:'))
divisor1=int(input('Enter the divisor:'))
try:
    num = dividend1/divisor1
    print(num)
except ZeroDivisionError:
    print("Cannot divide by zero")
finally:
    print("Execution completed.")
