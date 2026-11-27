from scypi.io import wavfile
import os

def cut_audio(path_in, path_out, beginning, end):

    samplerate, data = wavfile.read(path_in)
    index_beginning = beginning*samplerate
    index_end = end*samplerate

    if index_beginning < len(data) and index_end < len(data) :
        cut_audio = data[index_beginning:index_end]
        wavfile.write(path_out, samplerate, cut_audio)
    else:
        print("Audio "+path_in.split("/")[-2]+" - "+path_in.split("/")[-1]+" too short: ")
        print("Beginning: "+str(index_beginning))
        print("End: "+str(index_end))
        print("Length: "+str(len(data)))

orig_folder = "D:/04_Trimmed_audio/"
write_folder = "D:/06_Excerpt_audio/"

if not os.path.exists(write_folder) :
    os.mkdir(write_folder)

folders = os.listdir(orig_folder)

for folder in folders :

    files = os.listdir(orig_folder+"/"+folder)

    if not os.path.exists(write_folder+"/"+folder) :
        os.mkdir(write_folder+"/"+folder)

    for file in files :

        cut_audio(orig_folder+"/"+folder+"/"+file, write_folder+"/"+folder+"/"+file, 0, 10)
            
