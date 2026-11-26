import json
import os

from krajjat import Audio
from helper_functions import *

def pre_process_audio(subject, timestamps, verbosity=1):
    path_audio = op.join(PATH_AUDIO_CLEAN, subject + ".wav")
    audio_session = Audio(path_audio, verbosity=verbosity)

    for recording in timestamps[subject]:
        audio_recording = audio_session.trim(timestamps[subject][recording]["start"],
                                             timestamps[subject][recording]["end"],
                                             verbosity=verbosity)
        audio_recording.save(op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav"))

if __name__ == "__main__":
    with open("audio_timestamps.json") as f:
        timestamps = json.load(f)

    for subject in ALL_SUBJECTS:
        pre_process_audio(subject, timestamps)