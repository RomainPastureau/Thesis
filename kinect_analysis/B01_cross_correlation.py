import os
import pickle
import sys
from datetime import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.ticker as mticker
from constants import *

# Cross-correlation formula
def xcorr_pearson_matrix(series, reference, lags):

    # Coerce inputs to float arrays
    series = np.asarray(series, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if series.shape[1] != reference.size:
        raise ValueError(f"series and reference must have the same number of samples; "
                         f"got {series.shape[1]} and {reference.size}.")

    n_series, n_samples = series.shape
    correlations = np.full((n_series, len(lags)), np.nan, dtype=float)

    # For each lag, align the two signals to their overlap and compute Pearson r per series
    for i, lag in enumerate(lags):

        # Trim with lag
        series_aligned = series[:, max(-lag, 0):min(-lag, 0) or None]
        reference_aligned = reference[max(lag, 0):min(lag, 0) or None]

        n_overlap = reference_aligned.size
        if n_overlap < 10:
            continue

        sum_series = series_aligned.sum(axis=1)
        sum_reference = reference_aligned.sum()

        sum_series_sq = np.einsum("ij,ij->i", series_aligned, series_aligned)
        sum_reference_sq = np.dot(reference_aligned, reference_aligned)
        sum_cross = series_aligned @ reference_aligned

        numerator = sum_cross - (sum_series * sum_reference / n_overlap)

        series_ss = sum_series_sq - (sum_series * sum_series / n_overlap)
        reference_ss = sum_reference_sq - (sum_reference * sum_reference / n_overlap)
        denominator = np.sqrt(series_ss * reference_ss)

        valid = denominator > 0
        correlations[valid, i] = numerator[valid] / denominator[valid]

    return correlations


# Statistics
def test_r_at_lag0(per_subject_curves: list, lags_ms: np.ndarray, label: str) -> tuple[float, float]:
    """One-sample Wilcoxon signed-rank test: is r at lag=0 significantly different from 0?
    Returns (W statistic, p-value)."""
    lag0_idx = int(np.argmin(np.abs(lags_ms)))
    r_values = np.array([rs[lag0_idx] for rs in per_subject_curves])
    mean_r = float(np.mean(r_values))
    try:
        stat, p = wilcoxon(r_values)
    except ValueError:
        stat, p = np.nan, 1.0
    tick = "✅" if p < 0.05 else "❌"
    sig_str = "significant" if p < 0.05 else "not significant"
    print(f"  {tick} [{label}] r at lag=0: mean r = {mean_r:.4f}, Wilcoxon W = {stat:.1f}, p = {p:.4f} ({sig_str})")
    return float(stat), float(p)


def test_peak_lag_vs_0(per_subject_curves: list, lags_ms: np.ndarray, label: str) -> tuple[float, float]:
    """One-sample Wilcoxon signed-rank test: is the peak lag significantly different from 0?
    Returns (W statistic, p-value)."""
    peak_idxs = [int(np.argmax(rs)) for rs in per_subject_curves]
    peak_lags = np.array([lags_ms[i] for i in peak_idxs])
    peak_rs = np.array([float(per_subject_curves[s][i]) for s, i in enumerate(peak_idxs)])
    mean_lag = float(np.mean(peak_lags))
    sd_lag = float(np.std(peak_lags))
    se_lag = sd_lag / np.sqrt(len(peak_lags))
    mean_r = float(np.mean(peak_rs))
    try:
        stat, p = wilcoxon(peak_lags)
    except ValueError:
        stat, p = np.nan, 1.0
    tick = "✅" if p < 0.05 else "❌"
    sig_str = "significant" if p < 0.05 else "not significant"
    subj_lags_str = ", ".join([f"{float(l):.0f}" for l in peak_lags])
    subj_rs_str = ", ".join([f"{r:.4f}" for r in peak_rs])
    print(f"  {tick} [{label}] peak lag vs 0: mean = {mean_lag:.1f} ms (SD = {sd_lag:.1f}, SE = {se_lag:.1f}), "
          f"mean peak r = {mean_r:.4f}, Wilcoxon W = {stat:.1f}, p = {p:.4f} ({sig_str})")
    print(f"       Subject peak lags: [{subj_lags_str}] ms")
    print(f"       Subject peak r:    [{subj_rs_str}]")

# Plotting
def _draw_subpanel(ax: plt.Axes, subpanel: str, xcorr_joint_mean: dict, xcorr_group_mean: dict, xcorr_per_joint: dict,
                   xcorr_per_group: dict, lags_ms: np.ndarray, curve_labels=None) -> None:
    """Draw a single xcorr subpanel on ax.

    subpanel = "per_joint"   → group means in grey tones (Head: light → AllJoints: black)
    subpanel = joint name    → individual curves for that joint + their mean
    subpanel = group name    → individual curves for that group + their mean

    curve_labels: labels for the individual curves in the non-"per_joint" subpanels.
        None      → group figure: one curve per subject, coloured/labelled via SUBJECT_COLORS / SUBJECT_ANON.
        list[str] → per-subject figure: one curve per trial, labelled with these and coloured from a cycle.
    """
    if subpanel == "per_joint":
        # Group means in grey tones with SE bar at peak — no individual joint background
        for group, rs_mean in xcorr_group_mean.items():
            color = PER_JOINT_GREYS.get(group, "0.50")
            ax.plot(lags_ms, rs_mean, color=color, linewidth=2.2, label=group, zorder=4)
            if group in xcorr_per_group and len(xcorr_per_group[group]) > 1:
                peak_lags = np.array([lags_ms[np.argmax(s)] for s in xcorr_per_group[group]])
                se = float(np.std(peak_lags) / np.sqrt(len(peak_lags)))
                peak_idx = int(np.argmax(rs_mean))
                ax.errorbar(lags_ms[peak_idx], rs_mean[peak_idx], xerr=se, fmt="none", color=color, capsize=4,
                            linewidth=1.5, zorder=5)

        ax.set_title("Joint means across subjects", loc="center", fontsize=12, fontweight="bold")

    else:
        # Individual curves (per subject, or per trial in the per-subject figure) for a named joint or group
        if subpanel in xcorr_per_group and subpanel in xcorr_group_mean:
            individual_curves = xcorr_per_group[subpanel]
            rs_mean = xcorr_group_mean[subpanel]
        elif subpanel in xcorr_per_joint and subpanel in xcorr_joint_mean:
            individual_curves = xcorr_per_joint[subpanel]
            rs_mean = xcorr_joint_mean[subpanel]
        else:
            ax.set_visible(False)
            return

        # Resolve a label + colour for each individual curve
        if curve_labels is None:
            labels = [SUBJECT_ANON.get(s, s) for s in ALL_SUBJECTS]
            colors = [SUBJECT_COLORS.get(s, "grey") for s in ALL_SUBJECTS]
        else:
            labels = list(curve_labels)
            cmap = plt.get_cmap("tab10")
            colors = [cmap(k % 10) for k in range(len(individual_curves))]

        for curve, label, color in zip(individual_curves, labels, colors):
            ax.plot(lags_ms, curve, color=color, linewidth=1.2, alpha=0.8, label=label, zorder=3)
            peak_idx = int(np.argmax(curve))
            ax.plot(lags_ms[peak_idx], curve[peak_idx], "o", color=color, markersize=4, zorder=4)

        # Mean curve + SE bar at peak
        color_mean = PER_JOINT_GREYS.get(subpanel, "black")
        ax.plot(lags_ms, rs_mean, color=color_mean, linewidth=2.2, label="Mean", zorder=5)
        peak_lags = np.array([lags_ms[np.argmax(curve)] for curve in individual_curves])
        se = float(np.std(peak_lags) / np.sqrt(len(peak_lags)))
        peak_idx = int(np.argmax(rs_mean))
        ax.errorbar(lags_ms[peak_idx], rs_mean[peak_idx], xerr=se, fmt="none", color=color_mean, capsize=4,
                    linewidth=1.5, zorder=6)

        across = "trials" if curve_labels is not None else "subjects"
        ax.set_title(f"{subpanel} across {across}", loc="center", fontsize=12, fontweight="bold")


def plot_xcorr(xcorr_joint_mean: dict, xcorr_group_mean: dict, xcorr_per_joint: dict, xcorr_per_group: dict,
               lags_ms: np.ndarray, subpanels: list, kinematic_measure: str, acoustic_measure: str, out_path: str,
               show: bool, curve_labels=None) -> None:
    """Stack subpanels vertically and save the figure."""
    n = len(subpanels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, subpanel in zip(axes, subpanels):
        _draw_subpanel(ax, subpanel, xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group, lags_ms,
                       curve_labels=curve_labels)
        # ax.axvline(0, color="steelblue", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.set_xlim(lags_ms[0], lags_ms[-1])

    ax_bot = axes[-1]
    ax_bot.set_xlabel("Lag (ms)")
    ax_bot.text(0.0, -0.1, "Kinematics lead acoustics", transform=ax_bot.transAxes, ha="left", va="top", fontsize=9)
    ax_bot.text(1.0, -0.1, "Acoustics lead kinematics", transform=ax_bot.transAxes, ha="right", va="top", fontsize=9)

    axes[0].invert_xaxis()  # gesture-leads on the left (negative lags now shown first)
    axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(-x)}"))
    subj_range = f"{SUBJECT_ANON[ALL_SUBJECTS[0]]}–{SUBJECT_ANON[ALL_SUBJECTS[-1]]}"
    # fig.suptitle(f"{acoustic_measure}–{kinematic_measure} cross-correlation | {subj_range} (N={len(ALL_SUBJECTS)})",
    #              fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.split(out_path)[0], exist_ok=True)
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved → {out_path}")


def cross_correlation(subjects, kinematic_measures, acoustic_measures, lags, joints, joint_groups,
                      per_subject=True, re_referenced=True, subpanels=None, fs=FS, start_at=1, end_at=None):
    if isinstance(kinematic_measures, str):
        kinematic_measures = [kinematic_measures]
    if isinstance(acoustic_measures, str):
        acoustic_measures = [acoustic_measures]
    if subpanels is None:
        subpanels = ["per_joint", "Average of all joints"]

    trailing_file_name = ""
    if re_referenced:
        trailing_file_name += REREF_SUFFIX

    # `lags` are sample offsets (fed to the xcorr); `lags_ms` is the same axis in ms (used for plotting/stats)
    lags_ms = lags * (1000.0 / fs)

    # Output folders (both per-subject and group are written every run, so create all four)
    os.makedirs(op.join(PATH_RESULTS, "Cross-correlation", "Audio", "Per subject"), exist_ok=True)
    os.makedirs(op.join(PATH_RESULTS, "Cross-correlation", "Audio", "Group"), exist_ok=True)
    os.makedirs(op.join(PATH_PLOTS, "Cross-correlation", "Audio", "Per subject"), exist_ok=True)
    os.makedirs(op.join(PATH_PLOTS, "Cross-correlation", "Audio", "Group"), exist_ok=True)

    total = ((len(subjects) + 1) * len(kinematic_measures) * len(acoustic_measures) if per_subject
             else len(kinematic_measures) * len(acoustic_measures))
    if end_at is None:
        end_at = total

    joint_to_idx = {joint: idx for idx, joint in enumerate(joints)}
    i = 1
    ran = 0
    start_time = dt.now()

    for kinematic_measure in kinematic_measures:
        for acoustic_measure in acoustic_measures:

            df_path = op.join(PATH_DATAFRAME, "FS" + str(int(fs)),
                              f"{SYSTEM.lower()}_{kinematic_measure}_{acoustic_measure}{trailing_file_name}.gzip")
            df = pd.read_parquet(df_path)

            xcorr_per_joint = {j: [] for j in joints}
            xcorr_per_group = {g: [] for g in joint_groups}

            for subject in subjects:

                subject_data = df[df["subject"] == subject]

                if start_at <= i <= end_at:
                    print("=" * 60)
                    print(f"Cross-correlation {i} of {total}: {kinematic_measure} × {acoustic_measure} (for subject {subject}, fs: {fs_dataframe})")
                    print("=" * 60)

                    trial_joint_xcorr = {j: [] for j in joints}
                    trial_group_xcorr = {g: [] for g in joint_groups}

                    for trial in subject_data["trial"].unique():
                        print(f"\tTrial {trial}")
                        trial_data = subject_data[subject_data["trial"] == trial]

                        kinematic_data = trial_data[trial_data["measure"] == kinematic_measure]
                        kinematic_data_wide = kinematic_data.pivot(index="timestamp", columns="label", values="value")
                        kinematic_data_joints = kinematic_data_wide.reindex(columns=joints).sort_index()

                        # drop_duplicates guards a non-unique timestamp index (reindex would raise otherwise)
                        acoustic_data = trial_data[trial_data["measure"] == acoustic_measure]
                        acoustic_series = (acoustic_data.drop_duplicates("timestamp").set_index("timestamp")["value"]
                                           .reindex(kinematic_data_joints.index))

                        present = kinematic_data_joints.columns[kinematic_data_joints.notna().any(axis=0)]
                        valid_rows = acoustic_series.notna() & kinematic_data_joints[present].notna().all(axis=1)

                        x = kinematic_data_joints.loc[valid_rows].to_numpy(dtype=float).T
                        y = acoustic_series.loc[valid_rows].to_numpy(dtype=float)

                        trial_xcorr = xcorr_pearson_matrix(x, y, lags)
                        subj_joint_xcorr = {joint: trial_xcorr[joint_to_idx[joint]] for joint in joints}

                        for joint in joints:
                            trial_joint_xcorr[joint].append(trial_xcorr[joint_to_idx[joint]])
                        for group, joints_in_group in joint_groups.items():
                            idxs = [joint_to_idx[j] for j in joints_in_group if j in joint_to_idx]
                            if idxs:
                                trial_group_xcorr[group].append(np.nanmean(trial_xcorr[idxs], axis=0))

                    # Collapse this subject's trials into one mean curve per joint / group
                    subject_joint_mean = {j: np.nanmean(np.vstack(v), axis=0)
                                          for j, v in trial_joint_xcorr.items() if v}
                    subject_group_mean = {g: np.nanmean(np.vstack(v), axis=0)
                                          for g, v in trial_group_xcorr.items() if v}

                    # Accumulate the subject means for the across-subject (group) figure
                    for joint, curve in subject_joint_mean.items():
                        xcorr_per_joint[joint].append(curve)
                    for group, curve in subject_group_mean.items():
                        xcorr_per_group[group].append(curve)

                    # Per-subject figure: this subject's per-trial curves + the subject mean
                    if per_subject:
                        trial_labels = [f"Trial {t}" for t in subject_data["trial"].unique()]
                        figure_path = op.join(PATH_PLOTS, "Cross-correlation", "Audio", "Per subject",
                                              f"{kinematic_measure} x {acoustic_measure}",
                                              f"xcorr_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}_{subject}"
                                              f"{trailing_file_name}.png")
                        print(f"Saving the per-subject figure to {figure_path}...")
                        plot_xcorr(subject_joint_mean, subject_group_mean, trial_joint_xcorr, trial_group_xcorr,
                                   lags_ms, subpanels, kinematic_measure, acoustic_measure, figure_path,
                                   show=False, curve_labels=trial_labels)
                        ran += 1
                        i += 1
                        
                    if ran > 0:
                        print(f"Time elapsed: {dt.now() - start_time}")
                        print(f"Time remaining: {(dt.now() - start_time)/ran * (total - i)}")

            if start_at <= i <= end_at:
                print("=" * 60)
                print(f"Cross-correlation {i} of {total}: {kinematic_measure} × {acoustic_measure} (for all subjects, fs: {fs_dataframe})")
                print("=" * 60)

                # Across-subject means
                xcorr_joint_mean = {j: np.nanmean(np.vstack(v), axis=0) for j, v in xcorr_per_joint.items() if v}
                xcorr_group_mean = {g: np.nanmean(np.vstack(v), axis=0) for g, v in xcorr_per_group.items() if v}

                group_results_path = op.join(PATH_RESULTS, "Cross-correlation", "Audio", "Group",
                                            f"xcorr_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}"
                                            f"{trailing_file_name}.pkl")
                print(f"Saving group xcorr results to {group_results_path}...")
                with open(group_results_path, "wb") as f:
                    pickle.dump((xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group, lags_ms), f)

                group_figure_path = op.join(PATH_PLOTS, "Cross-correlation", "Audio", "Group",
                                            f"xcorr_{SYSTEM.lower()}_fs{int(fs_dataframe)}_{kinematic_measure}_{acoustic_measure}{trailing_file_name}.png")
                print(f"Saving the group figure to {group_figure_path}...")
                plot_xcorr(xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group, lags_ms, subpanels,
                        kinematic_measure, acoustic_measure, group_figure_path, show=False)
                ran += 1

            i += 1
            if ran > 0:
                print(f"Time elapsed: {dt.now() - start_time}")
                print(f"Time remaining: {(dt.now() - start_time)/ran * (total - i)}")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] not in ("velocity", "acceleration", "jerk", "snap"):
            print("Usage: python B07_cross_correlation.py [velocity|acceleration|jerk|snap] "
                  "[envelope|intensity|pitch|f1|f2]")
            sys.exit(1)
        kinematic_measures = [sys.argv[1]]
    else:
        kinematic_measures = KINEMATIC_MEASURES

    if len(sys.argv) > 2:
        if sys.argv[2] not in ("envelope", "intensity", "pitch", "f1", "f2"):
            print("Usage: python B07_cross_correlation.py [velocity|acceleration|jerk|snap] "
                  "[envelope|intensity|pitch|f1|f2]")
            sys.exit(1)
        acoustic_measures = [sys.argv[2]]
    else:
        acoustic_measures = ["envelope", "intensity", "pitch", "f1", "f2"]
    if len(sys.argv) > 3:
        fs_dataframe = int(sys.argv[3])
    else:
        fs_dataframe = FS

    re_referenced = True
    lag_max_ms = 10000
    lag_step_ms = 1000 / fs_dataframe
    lags = np.arange(-round(lag_max_ms / lag_step_ms), round(lag_max_ms / lag_step_ms) + 1, dtype=int)

    joint_groups = {"HandRight": ["HandRight"], "HandLeft": ["HandLeft"], "Head": ["Head"],
                    "Average of all joints": ALL_JOINTS}
    subpanels = ["per_joint", "Average of all joints"]

    start_at = 1
    end_at = None
    cross_correlation(ALL_SUBJECTS, kinematic_measures, acoustic_measures, lags, UPPER_BODY_JOINTS_REREF, joint_groups,
                      fs=fs_dataframe, per_subject=True, re_referenced=re_referenced, subpanels=subpanels, start_at=1, end_at=None)
