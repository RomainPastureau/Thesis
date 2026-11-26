import matplotlib
matplotlib.use("Agg")
print("MPL backend:", matplotlib.get_backend())

import os
import sys
from datetime import datetime as dt

import pandas as pd

from constants import *
from krajjat import analysis_functions as afunc


def coherence_joints(subjects, kinematic_measures, fs_dataframe, per_subject=True, exclude_joint=None, re_referenced=True, start_at=1,
                     average="trial", end_at=None, **kwargs):

    if isinstance(kinematic_measures, str):
        kinematic_measures = [kinematic_measures]

    if average == "trial":
        type_average = "_avgtrial"
    else:
        type_average = "_concat"

    trailing_file_name = ""
    if re_referenced:
        trailing_file_name += REREF_SUFFIX

    if per_subject:
        total = len(subjects) * len(kinematic_measures) * len(ALL_JOINTS)
        os.makedirs(op.join(PATH_RESULTS, "Coherence", "Joint", "Per subject"), exist_ok=True)
        os.makedirs(op.join(PATH_PLOTS, "Coherence", "Joint", "Per subject"), exist_ok=True)
    else:
        total = len(kinematic_measures) * len(ALL_JOINTS)
        os.makedirs(op.join(PATH_RESULTS, "Coherence", "Joint", "Group"), exist_ok=True)
        os.makedirs(op.join(PATH_PLOTS, "Coherence", "Joint", "Group"), exist_ok=True)
    i = 1
    ran = 0
    start_time = dt.now()

    if end_at is None:
        end_at = total

    for kinematic_measure in kinematic_measures:
        df = pd.read_parquet(
            op.join(PATH_DATAFRAME, "FS" + str(int(fs_dataframe)),
                    f"{SYSTEM.lower()}_{kinematic_measure}_envelope{trailing_file_name}.gzip"))

        for joint_label in ALL_JOINTS:

            if per_subject:
                for subject in subjects:

                    if start_at <= i <= end_at:
                        print("=" * 60)
                        print(f"Coherence {i} of {total}: {kinematic_measure} × {joint_label} (for subject {subject}, fs: {fs_dataframe})")
                        print("=" * 60)

                        path_png = op.join(PATH_PLOTS, "Coherence", "Joint", "Per subject", f"{kinematic_measure} x {joint_label}",
                                           f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{joint_label}_{subject}{type_average}{trailing_file_name}.png")
                        out = afunc.coherence(df, sequence_measure=kinematic_measure, audio_measure=None,
                                              coherence_with=joint_label, subjects=subject, random_seed=42, show=False,
                                              path_save=path_png, **kwargs)

                        out.save(op.join(PATH_RESULTS, "Coherence", "Joint", "Per subject", f"{kinematic_measure} x {joint_label}",
                                         f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{joint_label}_{subject}{type_average}{trailing_file_name}.gzip"),
                                 compact=True, compressed=True, include_plot_dictionary=False)
                        plt.close()

                        del out
                        ran += 1

                    i += 1

                    if ran > 0:
                        print(f"Time elapsed: {dt.now() - start_time}")
                        print(f"Time remaining: {(dt.now() - start_time)/ran * (total - i)}")

            else:

                if start_at <= i <= end_at:
                    print("=" * 60)
                    print(f"Coherence {i} of {total}: {kinematic_measure} × {joint_label} (for all subjects, fs: {fs_dataframe})")
                    print("=" * 60)

                    path_png = op.join(PATH_PLOTS, "Coherence", "Joint", "Group", kinematic_measure, 
                                       f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{joint_label}{type_average}{trailing_file_name}.png")
                    out = afunc.coherence(df, sequence_measure=kinematic_measure, audio_measure=None,
                                          coherence_with=joint_label, subjects=subjects, random_seed=42, show=False,
                                          path_save=path_png, **kwargs)

                    out.save(op.join(PATH_RESULTS, "Coherence", "Joint", "Group", kinematic_measure,
                                     f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{joint_label}{type_average}{trailing_file_name}.gzip"),
                             compact=True, compressed=True, include_plot_dictionary=False)
                    plt.close()

                    del out
                    ran += 1

                i += 1

                if ran > 0:
                    print(f"Time elapsed: {dt.now() - start_time}")
                    print(f"Time remaining: {(dt.now() - start_time)/ran * (total - i)}")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] not in ("velocity", "acceleration", "jerk", "snap"):
            print("Usage: python analyze_data.py [velocity|acceleration|jerk|snap] [envelope|intensity|pitch|f1|f2]")
            sys.exit(1)
        else:
            kinematic_measures = [sys.argv[1]]
    else:
        kinematic_measures = ["velocity", "acceleration", "jerk"]
    if len(sys.argv) > 2:
        fs_dataframe = int(sys.argv[2])
    else:
        fs_dataframe = FS

    re_referenced = True
    result_type = "z-scores"
    permutation_method = "shift"
    number_of_randperms = 1000
    freq_resolution_hz = 0.2
    lags = None
    n_jobs = 16

    start_at = 1550
    end_at = None
    coherence_joints(ALL_SUBJECTS_MINUS_S004, kinematic_measures, fs_dataframe, per_subject=True, exclude_joint=None, average="trial",
                     result_type=result_type, permutation_method=permutation_method, re_referenced=re_referenced,
                     number_of_randperms=number_of_randperms, freq_resolution_hz=freq_resolution_hz,
                     signif_alpha=0.05, signif_tail="1", lags=lags, signif_direction="up", joint_layout="auto",
                     n_jobs=n_jobs, start_at=start_at, end_at=end_at)

    start_at = 1
    end_at = None
    coherence_joints(ALL_SUBJECTS_MINUS_S004, kinematic_measures, fs_dataframe, per_subject=False, exclude_joint=None, series="subject",
                     average="subject", result_type=result_type, permutation_method=permutation_method,
                     re_referenced=re_referenced, number_of_randperms=number_of_randperms,
                     freq_resolution_hz=freq_resolution_hz, signif_alpha=0.05, signif_tail="1", lags=lags,
                     signif_direction="up", joint_layout="auto", n_jobs=n_jobs, start_at=start_at, end_at=end_at)
