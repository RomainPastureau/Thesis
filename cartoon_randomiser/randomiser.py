import os, random

f1 = open("input.txt", "r").read().split("\n")
g = open("randomized.txt", "w+")

random.shuffle(f1)

for i in range(len(f1)) :
    if i != len(f1)-1 :
        g.write(f1[i]+"\n")
    else :
        g.write(f1[i])

g.close()
