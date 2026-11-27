import os

def get_averages(folder, target):
    files = os.listdir(folder)

    averages = []
    for f in files :
        averages.append(get_average(folder+"/"+f))

    corrections = []
    for a in averages :
        corrections.append(get_correction(a, target))

    save_averages(folder, averages, corrections)

    save_corrections(folder, files, corrections)

def get_average(file):
    f = open(file, "r")
    fileread = f.read().split("\n")
    f.close()

    averages = []
    for i in range(1, len(fileread), 3):
        avg = float(fileread[i].split("=")[1])
        averages.append(avg)

    avg_avg = sum(averages)/len(averages)

    return(avg_avg)

def get_correction(average, target):
    correction = target/average
    return(correction)

def save_averages(folder, averages, corrections):

    f = open(folder+".txt", "w")

    for i in range(len(averages)) :
        f.write(str(averages[i])+"\t"+str(corrections[i])+"\n")

def save_corrections(folder, files, corrections):
    f = open("corrections.txt", "a")

    for i in range(len(corrections)):
         f.write(str(folder)+"\t"+str(files[i][0:8])+"\t"+str(corrections[i])+"\n")

    f.close()
        
target = 50
f = open("corrections.txt", "w")
f.close()

get_averages("K01", target)
get_averages("K02", target)
get_averages("K03", target)
get_averages("K04", target)
get_averages("K05", target)
get_averages("K06", target)
