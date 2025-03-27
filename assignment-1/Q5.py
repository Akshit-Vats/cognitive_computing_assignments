#5.1
L=[32,45,16,67,10]

smallest=min(L)
largest=max(L)

print(f'Largest and smallest numbers in the list are {smallest} and {largest} respectively')

#5.2
D={1:'One',2:'Two',3:'Three'}
print(D[1]) 

print(D.get(1))

#5.3
li=[23,12,32,43,54,15,78,1,23,4,56,6]
Ascending=sorted(li)
Descending=sorted(li,reverse='true')
Descending2=Ascending[::-1]

print('Ascending Order:',Ascending)

print('Descending Order:',Descending)

print('Descending Order 2:',Descending2)

#5.4
dict1={'Name':'akshit vats','Roll':102317091}
dict2={'Subject':'Cognitive Computing','Branch':'COPC'}
merged_dict={}
merged_dict.update(dict1)
merged_dict.update(dict2)
print(merged_dict)
