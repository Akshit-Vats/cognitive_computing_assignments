#6.1

string1=input('Enter a string: ')
vowels='aeiouAEIOU'
no_of_vowels=0
for i in string1:
    if i in vowels:
        no_of_vowels+=1

print(f'The number of vowels in {string1} are {no_of_vowels}')

#6.2

string2=input('Enter another string: ')
reversed_string=string2[::-1]
print(reversed_string)

#6.3

string3=input('Enter another string: ')
if string3==string3[::-1]:
    print("Palindrome")
else:
    print("Not a Palindrome")
