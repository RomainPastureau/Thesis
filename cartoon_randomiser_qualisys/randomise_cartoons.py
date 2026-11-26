import os, random, shutil, sys

def randomize(input_folder, output_folder):

    files = os.listdir(input_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    random.shuffle(files)
    i = 1
    for f in files:
        shutil.copy(input_folder+"/"+f, output_folder+"/R"+str(i).zfill(3)+"_"+f)
        i += 1

randomize("05 - Qualisys", "06 - Qualisys Randomized")
