import matplotlib
matplotlib.use("Agg")
print("MPL backend:", matplotlib.get_backend())

import os
import sys
from datetime import datetime as dt

import pandas as pd

from constants import *
from krajjat import analysis_functions as afunc

from C06_coherence_plotting import plot_coherence

custom_params = {"axes.spines.right": False, "axes.spines.top": False, "font.family": "Metropolis",
"xtick.labelsize": 8, "ytick.labelsize": 8}
sns_kwargs = {"style": "whitegrid", "rc": custom_params}
mpl_kwargs = {"hspace": 0.5, "tick_pad": -3, "xlabel_pad": 1, "ylabel_pad": 1,
              "title_pad": 3, "title_weight": "bold", "xtick_step": 1, "tight_layout": False}


def coherence_acoustic(subjects, kinematic_measures, acoustic_measures, fs_dataframe, per_subject=True, re_referenced=True,
                       average="trial", start_at=1, end_at=None, **kwargs):
    if isinstance(kinematic_measures, str):
        kinematic_measures = [kinematic_measures]
    if isinstance(acoustic_measures, str):
        acoustic_measures = [acoustic_measures]

    if average == "trial":
        type_average = "_avgtrial"
    else:
        type_average = "_concat"

    trailing_file_name = ""
    if re_referenced:
        trailing_file_name += REREF_SUFFIX

    if per_subject:
        total = len(subjects) * len(kinematic_measures) * len(acoustic_measures)
        os.makedirs(op.join(PATH_RESULTS, "Coherence", "Audio", "Per subject"), exist_ok=True)
        os.makedirs(op.join(PATH_PLOTS, "Coherence", "Audio", "Per subject"), exist_ok=True)
    else:
        total = len(kinematic_measures) * len(acoustic_measures)
        os.makedirs(op.join(PATH_RESULTS, "Coherence", "Audio", "Group"), exist_ok=True)
        os.makedirs(op.join(PATH_PLOTS, "Coherence", "Audio", "Group"), exist_ok=True)

    i = 1
    ran = 0
    start_time = dt.now()

    if end_at is None:
        end_at = total

    for kinematic_measure in kinematic_measures:
        for acoustic_measure in acoustic_measures:
            df = pd.read_parquet(
                op.join(PATH_DATAFRAME, "FS" + str(int(fs_dataframe)), 
                        f"{SYSTEM.lower()}_{kinematic_measure}_{acoustic_measure}{trailing_file_name}.gzip"))

            if per_subject:
                for subject in subjects:

                    if start_at <= i <= end_at:
                        print("=" * 60)
                        print(
                            f"Coherence {i} of {total}: {kinematic_measure} × {acoustic_measure} (for subject {subject}, fs: {fs_dataframe})")
                        print("=" * 60)

                        path_png = op.join(PATH_PLOTS, "Coherence", "Audio", "Per subject", f"{kinematic_measure} x {acoustic_measure}",
                                           f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}_{subject}{type_average}{trailing_file_name}.png")
                        
                        out = afunc.coherence(df, sequence_measure=kinematic_measure, audio_measure=acoustic_measure,
                                              coherence_with=acoustic_measure, subjects=subject, random_seed=42, average=average,
                                              show=False, path_save=path_png, **kwargs)

                        out.save(op.join(PATH_RESULTS, "Coherence", "Audio", "Per subject", f"{kinematic_measure} x {acoustic_measure}",
                                         f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}_{subject}{type_average}{trailing_file_name}.gzip"),
                                 compact=True, compressed=True, include_plot_dictionary=False)
                        
                        # Figure 1: 6 joints, subject, z-scores
                        plot_coherence(kinematic_measure, acoustic_measure, fs_dataframe, kind="z-scores", level=subject, re_referenced=re_referenced, 
                                       normalization=None, show_average="below", average=average,
                                       xlim=[0, 4], lags=0, show=False, legend=False,
                                       joint_layout=[["HandInRight", "HandInLeft", "HeadFront"], 
                                                        ["Chest", "WaistBackRight", "AnkleLeft"]],
                                       sns_kwargs=sns_kwargs, mpl_kwargs=mpl_kwargs,
                                       x_label = "Frequency (Hz)", y_label = "Coherence (z-scores)",
                                       show_peaks=False, show_peaks_sd=False, error_type="standard_error",
                                       path_png = op.join(PATH_PLOTS, "Coherence", "Audio", "Per subject", f"{kinematic_measure} x {acoustic_measure}",
                                                          f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}_{subject}{type_average}{trailing_file_name}_zscores.png"))
                        
                        # Figure 2: 6 joints, subject, average
                        plot_coherence(kinematic_measure, acoustic_measure, fs_dataframe, kind="average", level=subject, re_referenced=re_referenced, 
                                       normalization=None, show_average="below", average=average,
                                       xlim=[0, 4], lags=0, show=False, legend=False,
                                       joint_layout=[["HandInRight", "HandInLeft", "HeadFront"], 
                                                        ["Chest", "WaistBackRight", "AnkleLeft"]],
                                       sns_kwargs=sns_kwargs, mpl_kwargs=mpl_kwargs,
                                       x_label = "Frequency (Hz)", y_label = "Coherence (average)",
                                       show_peaks=False, show_peaks_sd=False, error_type="standard_error",
                                       path_png = op.join(PATH_PLOTS, "Coherence", "Audio", "Per subject", f"{kinematic_measure} x {acoustic_measure}",
                                                          f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}_{subject}{type_average}{trailing_file_name}_average.png"))
                        
                        # Figure 3: 6 joints, subject, z-scores
                        plot_coherence(kinematic_measure, acoustic_measure, fs_dataframe, kind="z-scores", level=subject, re_referenced=re_referenced, 
                                       normalization=None, show_average="below", average=average, x_axis="lags", frequencies=[1], xlim = [-1000, 1000],
                                       lags=lags, show=False, legend=False, ylim=[0, 20], 
                                       joint_layout=[["HandInRight", "HandInLeft", "HeadFront"], 
                                                        ["Chest", "WaistBackRight", "AnkleLeft"]],
                                       sns_kwargs=sns_kwargs, mpl_kwargs=mpl_kwargs,
                                       x_label = "Lag (ms)", y_label = "Coherence (z-scores)",
                                       show_peaks=False, show_peaks_sd=False, error_type="standard_error",
                                       path_png = op.join(PATH_PLOTS, "Coherence", "Audio", "Per subject", f"{kinematic_measure} x {acoustic_measure}",
                                                          f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}_{subject}{type_average}{trailing_file_name}_lags_zscores.png"))
                        
                        # Figure 4: 6 joints, subject, average
                        plot_coherence(kinematic_measure, acoustic_measure, fs_dataframe, kind="average", level=subject, re_referenced=re_referenced, 
                                       normalization=None, show_average="below", average=average, x_axis="lags", frequencies=[1], xlim = [-1000, 1000],
                                       lags=lags, show=False, legend=False, ylim=[0, 20],
                                       joint_layout=[["HandInRight", "HandInLeft", "HeadFront"], 
                                                        ["Chest", "WaistBackRight", "AnkleLeft"]],
                                       sns_kwargs=sns_kwargs, mpl_kwargs=mpl_kwargs,
                                       x_label = "Lag (ms)", y_label = "Coherence (average)",
                                       show_peaks=False, show_peaks_sd=False, error_type="standard_error",
                                       path_png = op.join(PATH_PLOTS, "Coherence", "Audio", "Per subject", f"{kinematic_measure} x {acoustic_measure}",
                                                          f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}_{subject}{type_average}{trailing_file_name}_lags_average.png"))
                        
                        del out
                        ran += 1

                    i += 1

                    if ran > 0:
                        print(f"Time elapsed: {dt.now() - start_time}")
                        print(f"Time remaining: {(dt.now() - start_time)/ran * (total - i)}")
                    
            else:

                if start_at <= i <= end_at:
                    print("=" * 60)
                    print(f"Coherence {i} of {total}: {kinematic_measure} × {acoustic_measure} (for all subjects, fs: {fs_dataframe})")
                    print("=" * 60)

                    path_png = op.join(PATH_PLOTS, "Coherence", "Audio", "Group",
                                       f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}{type_average}{trailing_file_name}.png")
                    out = afunc.coherence(df, sequence_measure=kinematic_measure, audio_measure=acoustic_measure,
                                          coherence_with=acoustic_measure, subjects=subjects, random_seed=42,
                                          show=False, path_save=path_png, **kwargs)

                    out.save(op.join(PATH_RESULTS, "Coherence", "Audio", "Group",
                                     f"coh_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}{type_average}{trailing_file_name}.gzip"),
                             compact=True, compressed=True, include_plot_dictionary=False)

                    del out
                    ran += 1

                i += 1

                if ran > 0:
                    print(f"Time elapsed: {dt.now() - start_time}")
                    print(f"Time remaining: {(dt.now() - start_time)/ran * (total - i)}")

                

if __name__ == "__main__":

    if len(sys.argv) == 1:
        kinematic_measures = KINEMATIC_MEASURES
    elif len(sys.argv) > 1:
        if sys.argv[1] not in ("velocity", "acceleration", "jerk", "snap"):
            print("Usage: python analyze_data.py [velocity|acceleration|jerk|snap] [envelope|intensity|pitch|f1|f2]")
            sys.exit(1)
        else:
            kinematic_measures = [sys.argv[1]]
    else:
        kinematic_measures = ["velocity", "acceleration", "jerk"]
    if len(sys.argv) > 2:
        if sys.argv[2] not in ("envelope", "intensity", "pitch", "f1", "f2"):
            print("Usage: python analyze_data.py [velocity|acceleration|jerk|snap] [envelope|intensity|pitch|f1|f2]")
            sys.exit(1)
        else:
            acoustic_measures = [sys.argv[2]]
    else:
        acoustic_measures = ["envelope", "intensity", "pitch", "f1", "f2"]
    if len(sys.argv) > 3:
        fs_dataframe = int(sys.argv[3])
    else:
        fs_dataframe = FS

    re_referenced = True
    result_type = "z-scores"
    permutation_method = "shift"
    number_of_randperms = 1000
    freq_resolution_hz = 0.2
    lags = [i / 1000 for i in range(-1000, 1040, 40)]
    n_jobs = 32

    start_at = 1
    end_at = None
    coherence_acoustic(ALL_SUBJECTS, kinematic_measures, acoustic_measures, fs_dataframe, per_subject=True, average="subject",
                       result_type=result_type, permutation_method=permutation_method, re_referenced=re_referenced,
                       number_of_randperms=number_of_randperms, freq_resolution_hz=freq_resolution_hz,
                       signif_alpha=0.05, signif_tail="1", lags=lags, signif_direction="up", joint_layout="auto",
                       n_jobs=n_jobs, start_at=start_at, end_at=end_at)

    # start_at = 1
    # end_at = None
    # coherence_acoustic(ALL_SUBJECTS_MINUS_S004, kinematic_measures, acoustic_measures, fs_dataframe, per_subject=False, series="subject",
    #                    average="subject", result_type=result_type, permutation_method=permutation_method,
    #                    re_referenced=re_referenced, number_of_randperms=number_of_randperms,
    #                    freq_resolution_hz=freq_resolution_hz, signif_alpha=0.05, signif_tail="1", lags=lags,
    #                    signif_direction="up", joint_layout="auto", n_jobs=n_jobs, start_at=start_at, end_at=end_at)
