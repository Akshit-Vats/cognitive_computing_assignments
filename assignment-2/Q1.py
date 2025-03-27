L=[10,20,30,40,50,60,70,80]

# (i)
L.append(200)
L.append(300)
print(L)

# (ii)
L.remove(10)
L.remove(30)
print(L)

# (iii)
L.sort()
print(L)

# (iii)

M=L[::-1]
print(M)

#or 
L.sort(reverse=True)
print(L)
