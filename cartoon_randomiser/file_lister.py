import os

file_list = os.listdir(".")

f = open("file_list.txt", "w+")

l = []

for file in file_list :
    l.append(str(file))

l.sort()

for i in range(len(l)) :
    f.write(l[i])
    if i != len(l)-1 :
        f.write("\n")

f.close()
        

