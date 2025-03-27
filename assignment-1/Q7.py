#7.1

with open('test.txt', 'w') as file:
    file.write("Hello, World!")

with open('test.txt', 'r') as file:
    content=file.read()
    print(content)

#7.2

with open('test.txt', 'a') as file:
    file.write("\nAppended text.")

with open('test.txt', 'r') as file:
    content=file.read()
    print(content)

#7.3

with open('test.txt', 'r') as file:
    lines = file.readlines()
    print("Number of lines:", len(lines))
