import os
from ffmpy import FFprobe

def get_luminance(origin, target):

    files = os.listdir(origin)
    inputs = {}
    for f in files :
        inputs[f] = "signalstats -show_entries frame_tags=lavfi.signalstats.YAVG"

    outputs = {}
    for f in files :
        outputs[target+"/"+f] = None

    ff = FFprobe(inputs=inputs)
    print(ff.cmd)

    ff.run()

get_luminance("01_Preprocessed_videos/K02",
              "02_Cropped_videos/K02")
    
    

