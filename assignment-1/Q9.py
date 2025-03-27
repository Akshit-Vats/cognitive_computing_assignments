import random 

#9.1

for i in range(0,5):
    print(random.randint(1, 100))

#9.2

n=random.randint(1, 100)
print(n)
flag=1
if n < 2:
        print('Not Prime')
        flag=0
else:
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            print('Not Prime')
            flag=0
        else:
            continue
if flag==1:
    print('Prime Number')

#9.3

dice_roll=random.randint(1,6)
print(f'The dice says: {dice_roll}')

#9.4

nums = [1, 2, 3, 4, 5]
random.shuffle(nums)
print(nums)

#9.5

items=["apple", "banana", "orange","guava"]
selected_item = random.choice(items)
print("Selected item:", selected_item)

#9.6

characters ='QWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*()qwertyuiopasdfghjklzxcvbnm'
length=int(input('Enter the length of the random password: '))
password=''.join(random.choice(characters) for i in range(length))
print("Random password:", password)

#9.7
suit=['Hearts', 'Diamonds', 'Clubs', 'Spades']
rank=['A','2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
chosen_card = f'{random.choice(rank)} of {random.choice(suit)}'
print(chosen_card)
