from bucket import *

a = [0.1,0.3,0.5,0.6,0.9]
c = np.digitize(a,[0.2, 0.5])
e = [[] for _ in range (3)]
for i in range (len(a)):
    e[c[i]].append(a[i])

print(e)