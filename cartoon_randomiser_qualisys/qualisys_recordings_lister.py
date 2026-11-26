import os

path = "Cartoons/"

subject = "S012"
jumps = []
j = 0

i = 0
folders = os.listdir(path)
for subfolder in folders:
    if subfolder.startswith("06 - Qualisys"):
        if subfolder.endswith(subject):
            files = os.listdir(path + subfolder + "/")
            for file in files:
                print(file[5:9])
                i += 1
                while j < len(jumps) and jumps[j] == i :
                    print("")
                    j += 1
