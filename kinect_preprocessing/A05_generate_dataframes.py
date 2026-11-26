import gc
from krajjat import Experiment, Subject, Trial, Sequence, Audio, Envelope, Pitch, Intensity, Formant
from helper_functions import *
from constants import *

import pandas as pd
import matplotlib.pyplot as plt


def generate_dataframe(kinematic_measure, audio_derivative, masked, force=False, re_reference_joint_label=None,
                       verbosity=1, **kwargs):
    experiment = Experiment(SYSTEM)

    if re_reference_joint_label:
        path_dataframe = op.join(PATH_DATAFRAME,
                                 f"{SYSTEM.lower()}_{kinematic_measure}_{audio_derivative}{REREF_SUFFIX}.gzip")
    else:
        path_dataframe = op.join(PATH_DATAFRAME, f"{SYSTEM.lower()}_{kinematic_measure}_{audio_derivative}.gzip")

    if not op.exists(path_dataframe) or force:

        for subject_name in ALL_SUBJECTS:
            print(subject_name)
            subject = Subject(subject_name)
            recordings = os.listdir(op.join(PATH_MOCAP_PREPROCESSED, subject_name))

            for recording in recordings:
                print(f"\t{recording}")

                if subject_name in EXCLUDED_RECORDINGS and op.splitext(recording)[0] in EXCLUDED_RECORDINGS[
                    subject_name]:
                    print(f"\t\tExcluded")

                else:
                    trial_id = op.splitext(recording)[0]
                    condition = masked[subject_name][trial_id]

                    path_sequence = op.join(PATH_MOCAP_PREPROCESSED, subject_name, trial_id + ".tsv")
                    path_derivative = op.join(PATH_AUDIO_DERIVATIVE, audio_derivative, subject_name, trial_id + ".pkl")

                    sequence = Sequence(path_sequence, verbosity=verbosity)
                    if re_reference_joint_label:
                        sequence = sequence.re_reference(re_reference_joint_label)

                    if audio_derivative == "envelope":
                        derivative = Envelope(path_derivative, verbosity=verbosity)
                    elif audio_derivative == "pitch":
                        derivative = Pitch(path_derivative, verbosity=verbosity)
                    elif audio_derivative == "intensity":
                        derivative = Intensity(path_derivative, verbosity=verbosity)
                    else:
                        derivative = Formant(path_derivative, formant_number=int(audio_derivative[1]),
                                             verbosity=verbosity)

                    trial = Trial(trial_id, condition, sequence, derivative)
                    subject.add_trial(trial, verbosity=verbosity)

            experiment.add_subject(subject)

        experiment.save_dataframe(path_dataframe, sequence_measure=kinematic_measure, audio_measure=audio_derivative,
                                  exclude_columns=["group"], verbosity=verbosity, **kwargs)


def generate_dataframes(kinematic_measures, audio_derivatives, force=False,
                        re_reference_joint_label=None, verbosity=1, **kwargs):
    masked = get_conditions()
    experiment = Experiment("Kinect")

    # Load recordings
    print("Loading sequences...")
    for subject_name in ALL_SUBJECTS:
        print(subject_name)
        subject = Subject(subject_name)
        recordings = os.listdir(op.join(PATH_MOCAP_PREPROCESSED, subject_name))

        for recording in recordings:
            print(f"\t{recording}")

            if subject_name in EXCLUDED_RECORDINGS and op.splitext(recording)[0] in EXCLUDED_RECORDINGS[subject_name]:
                print(f"\t\tExcluded")

            else:
                trial_id = op.splitext(recording)[0]
                condition = masked[subject_name][trial_id]

                path_sequence = op.join(PATH_MOCAP_PREPROCESSED, subject_name, trial_id + ".tsv")
                sequence = Sequence(path_sequence, verbosity=verbosity)
                if re_reference_joint_label:
                    sequence = sequence.re_reference(re_reference_joint_label)
                trial = Trial(trial_id, condition, sequence)
                subject.add_trial(trial, verbosity=verbosity)

        experiment.add_subject(subject)

    for audio_derivative in audio_derivatives:
        print(f"Loading {audio_derivative}...")
        for subject_name in ALL_SUBJECTS:
            print(subject_name)
            recordings = os.listdir(op.join(PATH_MOCAP_PREPROCESSED, subject_name))

            for recording in recordings:
                print(f"\t{recording}")

                if subject_name in EXCLUDED_RECORDINGS and op.splitext(recording)[0] in EXCLUDED_RECORDINGS[
                    subject_name]:
                    print(f"\t\tExcluded")

                else:
                    trial_id = op.splitext(recording)[0]
                    path_derivative = op.join(PATH_AUDIO_DERIVATIVE, audio_derivative, subject_name, trial_id + ".pkl")

                    if audio_derivative == "envelope":
                        derivative = Envelope(path_derivative, verbosity=verbosity)
                    elif audio_derivative == "pitch":
                        derivative = Pitch(path_derivative, verbosity=verbosity)
                    elif audio_derivative == "intensity":
                        derivative = Intensity(path_derivative, verbosity=verbosity)
                    else:
                        derivative = Formant(path_derivative, formant_number=int(audio_derivative[1]),
                                             verbosity=verbosity)
                    trial = experiment.get_subject(subject_name).get_trial(trial_id).set_audio(derivative)

        for kinematic_measure in kinematic_measures:
            if re_reference_joint_label:
                path_dataframe = op.join(PATH_DATAFRAME,
                                         f"kinect_{kinematic_measure}_{audio_derivative}{REREF_SUFFIX}.gzip")
            else:
                path_dataframe = op.join(PATH_DATAFRAME, f"kinect_{kinematic_measure}_{audio_derivative}.gzip")
            if not op.exists(path_dataframe) or force:
                print(f"Creating dataframe for {kinematic_measure} and {audio_derivative}...")
                experiment.save_dataframe(path_dataframe, sequence_measure=kinematic_measure,
                                          audio_measure=audio_derivative, exclude_columns=["group"],
                                          use_categoricals=True, verbosity=verbosity, **kwargs)

        # Unload audio before loading the next derivative
        print(f"Unloading {audio_derivative}...")
        for subject_name in experiment.subjects:
            for trial_id in experiment.subjects[subject_name].trials:
                experiment.subjects[subject_name].trials[trial_id].audio = None
        gc.collect()


if __name__ == "__main__":
    masked = get_conditions()

    # Single combination
    # generate_dataframe("velocity", "envelope", masked, force=True)

    # All combinations
    # for kinematic_measure in KINEMATIC_MEASURES:
    #     for audio_derivative in DERIVATIVES:
    #         generate_dataframe(kinematic_measure, audio_derivative, masked)

    # All combinations, faster
    kwargs = {"filter_below": CUTOFF_FREQ_LOW, "filter_above": None, "padtype": "odd", "padlen": 10 * FS}
    generate_dataframes(KINEMATIC_MEASURES, DERIVATIVES, force=True, **kwargs)
    generate_dataframes(KINEMATIC_MEASURES, DERIVATIVES, re_reference_joint_label="SpineBase", force=True, **kwargs)