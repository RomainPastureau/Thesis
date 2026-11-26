import os

from krajjat import Sequence, Audio
from krajjat import display_functions as kdisp
from krajjat import plot_functions as kplot

from helper_functions import *
from constants import *

def preprocess_sequence(subject, recording, delays, conditions, print_info=False, apply_low_pass=True, verbosity=1):

    # Paths
    if subject == "K06" and recording == "R013":
        path_mocap = op.join(PATH_MOCAP_ORIGINAL, subject, recording + "b.json")
    else:
        path_mocap = op.join(PATH_MOCAP_ORIGINAL, subject, recording + ".json")
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    path_mocap_out = op.join(PATH_MOCAP_PREPROCESSED, subject, recording + ".tsv")
    os.makedirs(op.dirname(path_mocap_out), exist_ok=True)

    # Recording info
    delay = delays[subject][recording]
    condition = conditions[subject][recording]

    # Load instances
    sequence = Sequence(path_mocap, name=recording, condition=condition, start_timestamps_at_zero=True,
                        verbosity=verbosity)
    audio = Audio(path_audio, name=recording, condition=condition, verbosity=verbosity)

    # Print info
    if print_info:
        print("\n== SEQUENCE ==")
        print(sequence.get_info("str"))
        print("\n== AUDIO ==")
        print(audio.get_info("str"))
        print(f"{subject} · {recording}")

    # Pre-processing
    sequence_cj = sequence.correct_jitter(0.1, 3, verbosity=verbosity)
    sequence_tr = sequence_cj.trim(-delay, audio.get_duration() - delay, verbosity=verbosity)
    sequence_tr.set_first_timestamp(0)
    sequence_rs = sequence_tr.resample(FS, verbosity=verbosity)
    if apply_low_pass:
        sequence_ff = sequence_rs.filter_frequencies(filter_above=CUTOFF_FREQ_HIGH, verbosity=verbosity)

    if print_info:
        print(f"\t\tPoses sequence: {sequence_rs.get_number_of_poses():,}, duration {sequence_rs.get_duration():.2f} s.".replace(",", " "))
        print(f"\t\tSamples audio: {audio.get_number_of_samples()}, duration {audio.get_duration():.2f} s.".replace(",", " "))
    if apply_low_pass:
        sequence_ff.save(path_mocap_out, verbosity=verbosity)
    else:
        sequence_rs.save(path_mocap_out, verbosity=verbosity)

def read_video(subject, recording, delays, verbosity=1):
    # Paths
    path_mocap = op.join(PATH_MOCAP_PREPROCESSED, subject, recording + ".tsv")
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    path_video = op.join(PATH_VIDEO, subject, recording + ".mp4")

    # Recording info
    delay = delays[subject][recording]

    # Load instances
    sequence = Sequence(path_mocap, name=recording, start_timestamps_at_zero=True, verbosity=verbosity)
    audio = Audio(path_audio, name=recording, verbosity=verbosity)

    kdisp.sequence_reader(sequence, path_audio, path_video, shift=VIDEO_SHIFTS[subject], zoom_level=2.4,
                          size_joint_default=20)

def compare_preprocessing(subject, recording, delays, verbosity=1):

    # Paths
    if subject == "K06" and recording == "R013":
        path_mocap = op.join(PATH_MOCAP_ORIGINAL, subject, recording + "b.json")
    else:
        path_mocap = op.join(PATH_MOCAP_ORIGINAL, subject, recording + ".json")
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    path_mocap_out = op.join(PATH_MOCAP_PREPROCESSED, subject, recording + ".tsv")
    path_video = op.join(PATH_VIDEO, subject, recording + ".mp4")

    # Recording info
    delay = delays[subject][recording]

    # Load instances
    sequence = Sequence(path_mocap, name=recording, start_timestamps_at_zero=True, verbosity=verbosity)
    audio = Audio(path_audio, name=recording, verbosity=verbosity)

    # Pre-processing
    sequence_cj = sequence.correct_jitter(0.1, 3, verbosity=verbosity)
    sequence_tr = sequence_cj.trim(-delay, audio.get_duration() - delay, verbosity=verbosity)
    sequence_tr.set_first_timestamp(0)
    sequence_rs = sequence_tr.resample(FS, verbosity=verbosity)
    sequence_ff = sequence_rs.filter_frequencies(filter_below=CUTOFF_FREQ_LOW, filter_above=CUTOFF_FREQ_HIGH,
                                                 verbosity=verbosity)

    kdisp.sequence_comparer(sequence_ff, sequence_rs, path_audio, path_video)
    kplot.single_joint_movement_plotter([sequence_ff, sequence_rs], "HandRight")

if __name__ == "__main__":
    delays = get_delays()
    conditions = get_conditions()

    apply_low_pass = True

    # All recordings
    for subject in ALL_SUBJECTS:
        path_mocap_subject = op.join(PATH_MOCAP_ORIGINAL, subject)
        recordings = os.listdir(path_mocap_subject)
        for recording in recordings:
            if not (op.splitext(recording)[0] in EXCLUDED_RECORDINGS.get(subject, [])) and not recording == "R013b.json":
                preprocess_sequence(subject, op.splitext(recording)[0], delays, conditions, False,
                                    apply_low_pass, verbosity=0)

    # Single recording
    # preprocess_sequence("K01", "R004", delays, conditions)
    # read_video("K01", "R004", delays)

    # Compare before/after
    # compare_preprocessing("K01", "R004", delays, verbosity=1)
