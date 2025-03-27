import sys

# 10.1

try:
    num1 = int(sys.argv[1])
    num2 = int(sys.argv[2])
    print(num1 * num2)
except IndexError:
    print("Error: Please provide two numbers as command-line arguments.")
except ValueError:
    print("Error: Please ensure both inputs are numbers.")

#10.2

string = sys.argv[1]
print(len(string))
