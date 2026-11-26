import os

from krajjat import Sequence, Audio
from krajjat import display_functions as kdisp
from krajjat import plot_functions as kplot

from helper_functions import *
from constants import *

def preprocess_sequence(subject, recording, info, print_info=False, verbosity=1):

    path_mocap = op.join(PATH_MOCAP_ORIGINAL, subject, recording + ".tsv")
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    path_mocap_out = op.join(PATH_MOCAP_PREPROCESSED, subject, recording + ".tsv")
    os.makedirs(op.dirname(path_mocap_out), exist_ok=True)

    # Recording info
    delay = info[subject][recording]["delay"]
    cartoon = info[subject][recording]["cartoon"]

    # Load instances
    sequence = Sequence(path_mocap, name=recording, start_timestamps_at_zero=True, verbosity=verbosity)
    if op.exists(path_audio):
        audio = Audio(path_audio, name=recording, verbosity=verbosity)

    # Print info
    if print_info:
        print("\n== SEQUENCE ==")
        print(sequence.get_info("str"))
        print("\n== AUDIO ==")
        if op.exists(path_audio):
            print(audio.get_info("str"))
        else:
            print("No audio.")
        print(f"{subject} · {recording}")

    # Pre-processing
    sequence_im = sequence.interpolate_missing_data("pchip", verbosity=verbosity)
    sequence_cj = sequence_im.correct_jitter(0.1, 3, verbosity=verbosity)
    if op.exists(path_audio):
        if delay is None:
            print(f"\t\tWarning: no delay found for {subject} · {recording}, skipping trim.")
            sequence_tr = sequence_cj
        else:
            sequence_tr = sequence_cj.trim(delay, audio.get_duration() + delay, verbosity=verbosity)
            sequence_tr.set_first_timestamp(0)
    else:
        sequence_tr = sequence_cj
    sequence_ff = sequence_tr.filter_frequencies(filter_above=CUTOFF_FREQ_HIGH, verbosity=verbosity)
    sequence_rs = sequence_ff.resample(FS, verbosity=verbosity)

    print(f"\t\tPoses sequence: {sequence_rs.get_number_of_poses():,}, duration {sequence_rs.get_duration():.2f} s.".replace(",", " "))
    if op.exists(path_audio):
        print(f"\t\tSamples audio: {audio.get_number_of_samples()}, duration {audio.get_duration():.2f} s.".replace(",", " "))
    else:
        print(f"\t\tNo audio")
    sequence_rs.save(path_mocap_out, verbosity=verbosity)

def read_video(subject, recording, info, verbosity=1):
    # Paths
    path_mocap = op.join(PATH_MOCAP_PREPROCESSED, subject, recording + ".tsv")
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    path_video = op.join(PATH_VIDEO_FRONT, subject, recording + ".mp4")

    # Load instances
    sequence = Sequence(path_mocap, name=recording, start_timestamps_at_zero=True, verbosity=verbosity)
    audio = Audio(path_audio, name=recording, verbosity=verbosity)

    kdisp.sequence_reader(sequence, path_audio, path_video, size_joint_default=20, x_axis='-x', y_axis='z')

def compare_preprocessing(subject, recording, delays, verbosity=1):

    path_mocap = op.join(PATH_MOCAP_ORIGINAL, subject, recording + ".tsv")
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    path_mocap_out = op.join(PATH_MOCAP_PREPROCESSED, subject, recording + ".tsv")
    path_video = op.join(PATH_VIDEO_FRONT, subject, recording + ".mp4")

    # Recording info
    delay = delays[subject][recording]

    # Load instances
    sequence = Sequence(path_mocap, name=recording, verbosity=verbosity)
    audio = Audio(path_audio, name=recording, verbosity=verbosity)

    # Pre-processing
    sequence_im = sequence.interpolate_missing_data("pchip", verbosity=verbosity)
    sequence_cj = sequence_im.correct_jitter(0.1, 3, verbosity=verbosity)
    sequence_tr = sequence_cj.trim(delay, audio.get_duration() + delay, verbosity=verbosity)
    sequence_tr.set_first_timestamp(0)
    sequence_ff = sequence_tr.filter_frequencies(filter_above=CUTOFF_FREQ_HIGH, verbosity=verbosity)
    sequence_rs = sequence_ff.resample(FS, verbosity=verbosity)

    kdisp.sequence_comparer(sequence_ff, sequence_rs, path_audio, path_video)
    kplot.single_joint_movement_plotter([sequence_ff, sequence_rs], "HandRight")

if __name__ == "__main__":
    info = get_info(ALL_SUBJECTS)

    # All recordings
    for subject in ALL_SUBJECTS:
        path_mocap_subject = op.join(PATH_MOCAP_ORIGINAL, subject)
        recordings = os.listdir(path_mocap_subject)
        for recording in recordings:
            preprocess_sequence(subject, op.splitext(recording)[0], info, verbosity=0)

    # Single recording
    preprocess_sequence("S001", "R001", info)
    read_video("S001", "R001", info)

    # Compare before/after
    compare_preprocessing("S001", "R001", info)