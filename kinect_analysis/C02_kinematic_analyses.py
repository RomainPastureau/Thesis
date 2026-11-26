import json
import math
import os
import re
import warnings
from glob import glob

import pandas as pd
import numpy as np
from krajjat import Graph
from scipy import stats
from scipy.signal import welch
from colorama import Fore, Style
import pingouin as pg
from krajjat.plot_functions import plot_silhouette, plot_body_graphs
from constants import *
from helper_functions import save_figure, print_section_title
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.formula.api as smf
from itertools import combinations, product
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

PATH_DATAFRAME_VELOCITY_F = op.join(PATH_DATAFRAME, f"FS{int(FS)} filtered", f"{SYSTEM.lower()}_velocity_envelope.gzip")
PATH_DATAFRAME_VELOCITY_REREF_F = op.join(PATH_DATAFRAME, f"FS{int(FS)} filtered", f"{SYSTEM.lower()}_velocity_envelope{REREF_SUFFIX}.gzip")
PATH_DATAFRAME_VELOCITY_U = op.join(PATH_DATAFRAME, f"FS{int(FS)} unfiltered", f"{SYSTEM.lower()}_velocity_envelope.gzip")
PATH_DATAFRAME_VELOCITY_REREF_U = op.join(PATH_DATAFRAME, f"FS{int(FS)} unfiltered", f"{SYSTEM.lower()}_velocity_envelope{REREF_SUFFIX}.gzip")
PATH_FIGURES = op.join(PATH_PLOTS, "Kinematics")
os.makedirs(PATH_FIGURES, exist_ok=True)

USE_REREF = True

# ── Coherence result paths (Group / Per subject, gzip layout) ────────────────
PATH_COHERENCE_GROUP = op.join(PATH_RESULTS, "Coherence", "Joint", "Group")
PATH_COHERENCE_SUBJECT = op.join(PATH_RESULTS, "Coherence", "Joint", "Per subject")


def _coherence_group_path(kinematic_measure, joint_label):
    """Path to the group-level (concatenated) coherence result for a given joint."""
    trailing_file_name = REREF_SUFFIX if USE_REREF else ""
    return op.join(PATH_COHERENCE_GROUP, kinematic_measure,
                   f"coh_{SYSTEM.lower()}_fs{int(FS)}_{kinematic_measure}_{joint_label}_concat{trailing_file_name}.gzip")


def _coherence_subject_dir(kinematic_measure, joint_label):
    """Directory holding the per-subject coherence results for a given joint."""
    return op.join(PATH_COHERENCE_SUBJECT, f"{kinematic_measure} x {joint_label}")


def _coherence_subject_path(kinematic_measure, joint_label, subject):
    """Path to a single subject's coherence result for a given joint."""
    trailing_file_name = REREF_SUFFIX if USE_REREF else ""
    return op.join(_coherence_subject_dir(kinematic_measure, joint_label),
                   f"coh_{SYSTEM.lower()}_fs{int(FS)}_{kinematic_measure}_{joint_label}_{subject}_avgtrial{trailing_file_name}.gzip")


def _coherence_subject_files(kinematic_measure, joint_label, subjects=None):
    """Return (subject_names, file_paths) found on disk for a given joint.

    If `subjects` is given, restrict to that list (in the given order) and only
    keep the ones that actually exist on disk.
    """
    if subjects is None:
        subjects = ALL_SUBJECTS
    found_names, found_files = [], []
    for subject in subjects:
        path = _coherence_subject_path(kinematic_measure, joint_label, subject)
        if op.exists(path):
            found_names.append(subject)
            found_files.append(path)
    return found_names, found_files


def _load_coherence_results(kinematic_measure, joint_label, level="subject", subjects=None):
    """Load coherence Results object(s) for a joint under the Kinect gzip layout.

    level="group"   → single Results object (concatenated group analysis), or None if missing.
    level="subject" → dict {subject: Results}, one entry per subject file found on disk.
    """
    from krajjat import Results

    if level == "group":
        path = _coherence_group_path(kinematic_measure, joint_label)
        if not op.exists(path):
            print(f"  No group file found:\n  {path}")
            return None
        return Results.load(path)

    elif level == "subject":
        subject_names, files = _coherence_subject_files(kinematic_measure, joint_label, subjects)
        if not files:
            print(f"  No subject files found for {kinematic_measure} × {joint_label}\n"
                  f"  in {_coherence_subject_dir(kinematic_measure, joint_label)}")
            return {}
        return {subject: Results.load(f) for subject, f in zip(subject_names, files)}

    else:
        raise ValueError(f"level must be 'group' or 'subject', got {level!r}")


def _extract_by_joint(out, target_joint, sequence_measure, value_type="z-score"):
    """Pull (data_by_joint, p_by_joint, lag, freqs) out of a Results object.

    value_type="z-score" → data_by_joint = out.z_scores[...], p_by_joint = out.p_values[...]
    value_type="raw"     → data_by_joint = out.averages[...] (raw coherence), p_by_joint = None
    """
    key = (target_joint, sequence_measure)
    sm_key = out.sequence_measures[0]
    sv = out.series_values[0]
    lag = out.lags[0]
    freqs = np.array(out.frequencies)

    if value_type == "z-score":
        data_by_joint = out.z_scores[key][sm_key][sv]
        p_by_joint = out.p_values[key][sm_key][sv]
    elif value_type == "raw":
        data_by_joint = out.averages[key][sm_key][sv]
        p_by_joint = None
    else:
        raise ValueError(f"value_type must be 'raw' or 'z-score', got {value_type!r}")

    return data_by_joint, p_by_joint, lag, freqs


def darken(color, factor=0.6):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))

def lighten(hex_color, factor=0.65):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * factor)


# Kendall's W on the full rank matrix
def kendalls_w(rank_matrix):
    # rank_matrix: subjects x joints
    n_raters, n_targets = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12 * S / (n_raters ** 2 * (n_targets ** 3 - n_targets))
    chi2 = n_raters * (n_targets - 1) * W
    p = stats.chi2.sf(chi2, df=n_targets - 1)
    return W, chi2, p


def mean_speed_by_joint(n, df, generate_figure=True, path_save=None):
    print_section_title(f"{n}. MEAN SPEED BY JOINT")

    joints = list(df["label"].unique())
    joints = [j for j in joints if j not in LOWER_BODY_JOINTS]
    mean_speed = df.groupby("label")["value"].mean().reindex(joints)
    print("\nOverall mean speed per joint:")
    print(mean_speed.round(4).to_string())

    # Build subjects x joints matrix of per-subject mean speeds
    mean_speed_subj = (df.groupby(["subject", "label"])["value"]
                       .mean().unstack("label").reindex(columns=joints))

    # Friedman test: do joints differ in speed? (subjects = blocks, joints = conditions)
    if USE_REREF:
        joints_for_test = [j for j in joints if j != "SpineBase"]  # exclude re-reference joint
    else:
        joints_for_test = joints.copy()
    friedman_data = [mean_speed_subj[j].values for j in joints_for_test]
    chi2, p_friedman = stats.friedmanchisquare(*friedman_data)
    print(f"\nFriedman test (do joints differ in mean speed?):")
    print(f"  chi2({len(joints_for_test) - 1}) = {chi2:.3f}, p = {p_friedman:.4f}")

    if generate_figure:
        plot_dict_overall = {joint: float(mean_speed[joint] * 100) for joint in joints}
        plot_silhouette(plot_dict_overall,
                        # title="Overall mean speed per joint",
                        title_scale="Mean speed (cm/s)",
                        min_scale=0, max_scale=float(mean_speed.max() * 100),
                        color_scheme="celsius",
                        path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"),
                        show=False,
                        path_save=path_save or op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_mean_speed_by_joint.png"),
                        resolution=(600, 1080),
                        show_values=True,
                        value_format=".1f",
                        font_size=0.8,
                        verbosity=0)


def mean_speed_comparison_between_joints(n, df, generate_figure=True):
    print_section_title(f"{n}. COMPARISON OF THE MEAN SPEED BETWEEN JOINTS")

    joints = list(df["label"].unique())
    mean_speed = df.groupby("label")["value"].mean().reindex(joints)

    # Friedman test: do joints differ in speed? (subjects = blocks, joints = conditions)
    if USE_REREF:
        joints_for_test = [j for j in joints if j != "SpineBase"]  # exclude re-reference joint
    else:
        joints_for_test = joints.copy()

    # Build subjects x joints matrix of per-subject mean speeds
    mean_speed_subj = (df.groupby(["subject", "label"])["value"]
                       .mean().unstack("label").reindex(columns=joints))

    # Drop subjects with missing data for any joint before the Friedman test
    friedman_matrix = mean_speed_subj[joints_for_test].copy()
    col_means = friedman_matrix.mean()  # ignores NaN by default
    friedman_matrix = friedman_matrix.fillna(col_means)
    n_complete = len(friedman_matrix)
    n_total = len(mean_speed_subj)

    # Pairwise Wilcoxon signed-rank tests across all joint pairs
    pair_results = []
    for j1, j2 in combinations(joints_for_test, 2):
        # Drop subjects missing either joint
        paired = friedman_matrix[[j1, j2]].dropna()
        stat, p = stats.wilcoxon(paired[j1], paired[j2])
        pair_results.append({"joint1": j1, "joint2": j2, "W": stat, "p": p})

    df_pairs = pd.DataFrame(pair_results)

    # FDR correction
    reject, p_adj, _, _ = multipletests(df_pairs["p"], method="fdr_bh")
    df_pairs["p_adj"] = p_adj
    df_pairs["significant"] = reject

    n_sig = reject.sum()
    print(f"\nPairwise Wilcoxon signed-rank tests (FDR-corrected, {len(df_pairs)} pairs):")
    print(f"  Significant pairs: {n_sig}/{len(df_pairs)}")
    print(df_pairs.to_string(index=False))

    if generate_figure:
        # Build significance matrix (NaN on diagonal, p_adj elsewhere)
        sig_matrix = pd.DataFrame(np.nan, index=joints_for_test, columns=joints_for_test)
        for _, row in df_pairs.iterrows():
            for j in joints_for_test:
                sig_matrix.loc[j, j] = 1.0
            sig_matrix.loc[row["joint1"], row["joint2"]] = row["p_adj"]
            sig_matrix.loc[row["joint2"], row["joint1"]] = row["p_adj"]

        # # Sort joints by overall mean speed (descending) for a cleaner gradient
        # joint_order = mean_speed_subj[joints_for_test].mean().sort_values(ascending=False).index.tolist()
        # sig_matrix = sig_matrix.reindex(index=joint_order, columns=joint_order)

        n_joints = len([j for j in joints if j != "SpineBase"])

        nonsig_mask = (sig_matrix >= 0.05)
        sig_mask = (sig_matrix < 0.05)

        fig, ax = plt.subplots(figsize=(16, 14))

        # Custom colormaps
        cmap_sig = LinearSegmentedColormap.from_list(
            "sig", ["#1a5c1a", "#8fd400"], N=256)  # dark green → sheen green
        cmap_nonsig = LinearSegmentedColormap.from_list(
            "nonsig", ["#e87000", "#cc0000", "#4b0082"], N=256)  # orange → red → purple

        # Green layer: significant pairs
        sns.heatmap(sig_matrix, mask=nonsig_mask, ax=ax,
                    cmap=cmap_sig, vmin=0, vmax=0.05,
                    linewidths=0.3, linecolor="white",
                    cbar=False, square=True,
                    xticklabels=True, yticklabels=True)

        # Orange-red-purple layer: non-significant pairs
        sns.heatmap(sig_matrix, mask=sig_mask, ax=ax,
                    cmap=cmap_nonsig, vmin=0.05, vmax=1.0,
                    linewidths=0.3, linecolor="white",
                    cbar=False, square=True,
                    xticklabels=True, yticklabels=True)

        # Annotate with p-values, white text throughout
        for row_i in range(n_joints):
            for col_i in range(n_joints):
                p_val = sig_matrix.iloc[row_i, col_i]
                if not np.isnan(p_val):
                    if p_val >= 0.995:
                        label = "1.000"
                    else:
                        label = f"{p_val:.3f}"[1:]  # ".03" format
                    ax.text(col_i + 0.5, row_i + 0.5, label,
                            ha="center", va="center",
                            fontsize=9, color="white")

        ax.set_title(
            "Pairwise Wilcoxon signed-rank tests — mean speed per joint\n"
            f"(FDR-corrected, joints ordered by mean speed)\n"
            r"$\bf{Orange-Red-Purple\ gradient}$" + " = non-significant (similar mean speed)  |  " +
            r"$\bf{Green\ gradient}$" + " = significant (different mean speed)",
            fontsize=11, pad=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

        plt.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_mean_speed_comparison_between_joints.png", **SAVE_KW)
        plt.close(fig)


def mean_speed_by_subject(n, df, generate_figure=True, ax=None):
    print_section_title(f"{n}. MEAN SPEED BY SUBJECT")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    # Overall mean speed per subject (collapsed across joints and trials)
    overall_mean = df_test.groupby("subject")["value"].mean()
    print("\nOverall mean speed per subject (across all joints):")
    overall_mean = overall_mean * 100
    print(overall_mean.round(2).to_string())
    print("")
    print(f"Min: {np.round(np.min(overall_mean), 2)}")
    print(f"Max: {np.round(np.max(overall_mean), 2)}")
    print(f"Mean of means: {np.round(np.mean(overall_mean), 2)}")
    print(f"SD of means: {np.round(np.std(overall_mean), 2)}")

    # Per-trial mean speed per subject (for stats and dot overlay)
    trial_means = (df_test.groupby(["subject", "trial"])["value"]
                   .mean().reset_index())
    trial_means.columns = ["subject", "trial", "mean_speed"]
    groups = [g["mean_speed"].values for _, g in trial_means.groupby("subject")]

    # ── Stats ──────────────────────────────────────────────────────────────────
    print("\nNormality tests (Shapiro-Wilk) on per-trial mean speed by subject:")
    normality_results = {}
    for subject, g in trial_means.groupby("subject"):
        stat, p = stats.shapiro(g["mean_speed"])
        normality_results[subject] = p
        sym = "✓" if p > 0.05 else "✗"
        print(f"  {subject}: W = {stat:.3f}, p = {p:.4f} {sym}")
    all_normal = all(p > 0.05 for p in normality_results.values())

    if all_normal:
        lev_stat, lev_p = stats.levene(*groups)
        equal_var = lev_p > 0.05
        print(f"\nLevene's test: W = {lev_stat:.3f}, p = {lev_p:.4f} "
              f"({'equal' if equal_var else 'unequal'} variances)")

    if all_normal and equal_var:
        f_stat, p_main = stats.f_oneway(*groups)
        print(f"\nOne-way ANOVA: F = {f_stat:.3f}, p = {p_main:.4f}")
        test_name = "Tukey HSD"
    elif all_normal:
        welch = pg.welch_anova(data=trial_means, dv="mean_speed", between="subject")
        p_main = welch["p_unc"].values[0]
        print(f"\nWelch's ANOVA: F = {welch['F'].values[0]:.3f}, p = {p_main:.4f}")
        test_name = "Games-Howell"
    else:
        h_stat, p_main = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis: H = {h_stat:.3f}, p = {p_main:.4f}")
        test_name = "Dunn"

    # ── Post-hoc + CLD ─────────────────────────────────────────────────────────
    cld = {s: "" for s in ALL_SUBJECTS}
    if p_main < 0.05:
        print(f"\nPost-hoc: {test_name}")

        if test_name == "Tukey HSD":
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            posthoc = pairwise_tukeyhsd(trial_means["mean_speed"],
                                        trial_means["subject"], alpha=0.05)
            print(posthoc)
            # Build p-value matrix from Tukey results
            subj_list = sorted(trial_means["subject"].unique())
            p_matrix = pd.DataFrame(np.ones((len(subj_list), len(subj_list))),
                                    index=subj_list, columns=subj_list)
            for row in posthoc._results_table.data[1:]:
                g1, g2, _, _, p, _, _ = row
                p_matrix.loc[g1, g2] = p
                p_matrix.loc[g2, g1] = p
        elif test_name == "Games-Howell":
            gh = pg.pairwise_gameshowell(data=trial_means, dv="mean_speed",
                                         between="subject")
            print(gh.to_string())
            subj_list = sorted(trial_means["subject"].unique())
            p_matrix = pd.DataFrame(np.ones((len(subj_list), len(subj_list))),
                                    index=subj_list, columns=subj_list)
            for _, row in gh.iterrows():
                p_matrix.loc[row["A"], row["B"]] = row["pval"]
                p_matrix.loc[row["B"], row["A"]] = row["pval"]
        else:  # Dunn
            import scikit_posthocs as sp
            p_matrix = sp.posthoc_dunn(trial_means, val_col="mean_speed",
                                       group_col="subject", p_adjust="holm")
            subj_ordered = overall_mean.sort_values().index.tolist()
            p_matrix = p_matrix.reindex(index=subj_ordered, columns=subj_ordered)
            print(p_matrix.to_csv(sep="\t"))

        # Build CLD: assign letters so that subjects sharing a letter are non-sig different
        subj_ordered = overall_mean.sort_values().index.tolist()  # slowest → fastest
        letter_idx = 0
        letter = chr(ord("a") + letter_idx)
        cld[subj_ordered[0]] = letter
        for i in range(1, len(subj_ordered)):
            s = subj_ordered[i]
            # Check against all subjects that share each current letter
            new_letters = set()
            for existing_letter in set("".join(cld.values())):
                group = [x for x in subj_ordered[:i] if existing_letter in cld[x]]
                # If not significantly different from all members of this group → share letter
                if all(p_matrix.loc[s, member] >= 0.05 for member in group):
                    new_letters.add(existing_letter)
            if not new_letters:
                letter_idx += 1
                new_letter = chr(ord("a") + letter_idx)
                new_letters.add(new_letter)
            cld[s] = "".join(sorted(new_letters))

        print("\nCompact letter display (subjects ordered by mean speed):")
        for s in subj_ordered:
            print(f"  {s}: {overall_mean[s]:.4f} m/s  [{cld[s]}]")

    # ── Figure ─────────────────────────────────────────────────────────────────
    if generate_figure:
        subj_sorted = overall_mean.sort_values(ascending=True).index.tolist()
        x_pos = {s: i for i, s in enumerate(subj_sorted)}

        external_ax = ax is not None
        if not external_ax:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Build a DataFrame with cm/s for boxplot
        trial_means_cm = trial_means.copy()
        trial_means_cm["mean_speed"] = trial_means_cm["mean_speed"] * 100

        bp = sns.boxplot(data=trial_means_cm, x="subject", y="mean_speed",
                         order=subj_sorted, width=0.5, fliersize=0, ax=ax,
                         color="white")

        for i, subj in enumerate(subj_sorted):
            color = SUBJECT_COLORS[subj]
            dark = darken(color)
            ax.patches[i].set_facecolor(color)
            ax.patches[i].set_edgecolor(dark)
            ax.patches[i].set_linewidth(1.5)
            bp.lines[i * 6 + 4].set_color(dark)
            bp.lines[i * 6 + 4].set_linewidth(2)

        for patch, subj in zip(ax.patches[:len(subj_sorted)], subj_sorted):
            color = SUBJECT_COLORS[subj]
            patch.set_facecolor(color)
            patch.set_edgecolor(darken(color))
            patch.set_linewidth(1.5)

        for line, subj in zip(bp.lines[4::6], subj_sorted):
            line.set_color(darken(SUBJECT_COLORS[subj]))
            line.set_linewidth(2)

        # Per-trial dots
        for i, subj in enumerate(subj_sorted):
            vals = trial_means_cm[trial_means_cm["subject"] == subj]["mean_speed"].values
            color = SUBJECT_COLORS[subj]
            ax.scatter(np.random.uniform(-0.2, 0.2, len(vals)) + i, vals,
                       color=lighten(color, factor=0.8), edgecolors=darken(color, factor=0.8),
                       linewidths=0.4, alpha=0.6, s=12, zorder=3)

        # ── CLD grouping lines ────────────────────────────────────────────────
        y_top = trial_means_cm.groupby("subject")["mean_speed"].mean().max()
        y_range = trial_means_cm["mean_speed"].max()
        base_h = y_top + y_range * 0.10
        step_h = y_range * 0.07
        tick_h = y_range * 0.015

        all_letters = sorted(set("".join(cld.values())))
        for li, letter in enumerate(all_letters):
            group = [s for s in subj_sorted if letter in cld[s]]
            xs = [x_pos[s] for s in group]
            x_left, x_right = min(xs) - 0.25, max(xs) + 0.25
            h = base_h + li * step_h
            if len(group) == 1:
                ax.plot([xs[0], xs[0]], [h - tick_h, h], color="black", linewidth=1.5, zorder=4)
            else:
                ax.plot([x_left, x_right], [h, h], color="black", linewidth=1.5,
                        zorder=4, solid_capstyle="round")
                for xend in [x_left, x_right]:
                    ax.plot([xend, xend], [h - tick_h, h], color="black", linewidth=1.5, zorder=4)
            ax.text((x_left + x_right) / 2, h + tick_h * 0.5, letter.upper(),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        top_line = base_h + (len(all_letters) - 1) * step_h + y_range * 0.08
        ax.set_ylim(0, top_line)
        ax.set_xticks(range(len(subj_sorted)))
        ax.set_xticklabels([SUBJECT_ANON[s] for s in subj_sorted], fontsize=10)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Mean speed (cm/s)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
        ax.set_axisbelow(True)

        if not external_ax:
            plt.tight_layout()
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_mean_speed_by_subject.png", **SAVE_KW)
            plt.close(fig)


def mean_speed_by_subject_joint(n, df, joint, generate_figure=True):
    print_section_title(f"{n}. MEAN SPEED BY SUBJECT — {joint.upper()}")

    df_joint = df[df["label"] == joint].copy()
    if df_joint.empty:
        print(f"  No data found for joint '{joint}'.")
        return

    # Overall mean speed per subject (collapsed across trials)
    overall_mean = df_joint.groupby("subject")["value"].mean()
    overall_mean = overall_mean * 100
    print(f"\nOverall mean speed per subject for {joint}:")
    print(overall_mean.round(4).to_string())
    print("")
    print(f"Min: {np.round(np.min(overall_mean), 2)}")
    print(f"Max: {np.round(np.max(overall_mean), 2)}")
    print(f"Mean of means: {np.round(np.mean(overall_mean), 2)}")
    print(f"SD of means: {np.round(np.std(overall_mean), 2)}")

    # Per-trial mean speed per subject (for stats and dot overlay)
    trial_means = (df_joint.groupby(["subject", "trial"])["value"]
                   .mean().reset_index())
    trial_means.columns = ["subject", "trial", "mean_speed"]
    groups = [g["mean_speed"].values for _, g in trial_means.groupby("subject")]

    # ── Stats ──────────────────────────────────────────────────────────────────
    print(f"\nNormality tests (Shapiro-Wilk) on per-trial mean speed by subject:")
    normality_results = {}
    for subject, g in trial_means.groupby("subject"):
        stat, p = stats.shapiro(g["mean_speed"])
        normality_results[subject] = p
        sym = "✓" if p > 0.05 else "✗"
        print(f"  {subject}: W = {stat:.3f}, p = {p:.4f} {sym}")
    all_normal = all(p > 0.05 for p in normality_results.values())

    equal_var = False
    if all_normal:
        lev_stat, lev_p = stats.levene(*groups)
        equal_var = lev_p > 0.05
        print(f"\nLevene's test: W = {lev_stat:.3f}, p = {lev_p:.4f} "
              f"({'equal' if equal_var else 'unequal'} variances)")

    if all_normal and equal_var:
        f_stat, p_main = stats.f_oneway(*groups)
        print(f"\nOne-way ANOVA: F = {f_stat:.3f}, p = {p_main:.4f}")
        test_name = "Tukey HSD"
    elif all_normal:
        welch = pg.welch_anova(data=trial_means, dv="mean_speed", between="subject")
        p_main = welch["p_unc"].values[0]
        print(f"\nWelch's ANOVA: F = {welch['F'].values[0]:.3f}, p = {p_main:.4f}")
        test_name = "Games-Howell"
    else:
        h_stat, p_main = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis: H = {h_stat:.3f}, p = {p_main:.4f}")
        test_name = "Dunn"

    # ── Post-hoc + CLD ─────────────────────────────────────────────────────────
    cld = {s: "" for s in subjects}
    if p_main < 0.05:
        print(f"\nPost-hoc: {test_name}")

        if test_name == "Tukey HSD":
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            posthoc = pairwise_tukeyhsd(trial_means["mean_speed"],
                                        trial_means["subject"], alpha=0.05)
            print(posthoc)
            subj_list = sorted(trial_means["subject"].unique())
            p_matrix = pd.DataFrame(np.ones((len(subj_list), len(subj_list))),
                                    index=subj_list, columns=subj_list)
            for row in posthoc._results_table.data[1:]:
                g1, g2, _, _, p, _, _ = row
                p_matrix.loc[g1, g2] = p
                p_matrix.loc[g2, g1] = p
        elif test_name == "Games-Howell":
            gh = pg.pairwise_gameshowell(data=trial_means, dv="mean_speed",
                                         between="subject")
            print(gh.to_string())
            subj_list = sorted(trial_means["subject"].unique())
            p_matrix = pd.DataFrame(np.ones((len(subj_list), len(subj_list))),
                                    index=subj_list, columns=subj_list)
            for _, row in gh.iterrows():
                p_matrix.loc[row["A"], row["B"]] = row["pval"]
                p_matrix.loc[row["B"], row["A"]] = row["pval"]
        else:  # Dunn
            import scikit_posthocs as sp
            p_matrix = sp.posthoc_dunn(trial_means, val_col="mean_speed",
                                       group_col="subject", p_adjust="holm")
            subj_ordered = overall_mean.sort_values().index.tolist()
            p_matrix = p_matrix.reindex(index=subj_ordered, columns=subj_ordered)
            print(p_matrix.to_csv(sep="\t"))

        subj_ordered = overall_mean.sort_values().index.tolist()
        letter_idx = 0
        letter = chr(ord("a") + letter_idx)
        cld[subj_ordered[0]] = letter
        for i in range(1, len(subj_ordered)):
            s = subj_ordered[i]
            new_letters = set()
            for existing_letter in set("".join(cld.values())):
                group = [x for x in subj_ordered[:i] if existing_letter in cld[x]]
                if all(p_matrix.loc[s, member] >= 0.05 for member in group):
                    new_letters.add(existing_letter)
            if not new_letters:
                letter_idx += 1
                new_letter = chr(ord("a") + letter_idx)
                new_letters.add(new_letter)
            cld[s] = "".join(sorted(new_letters))

        print(f"\nCompact letter display (subjects ordered by mean speed):")
        for s in subj_ordered:
            print(f"  {s}: {overall_mean[s]:.4f} m/s  [{cld[s]}]")

    # ── Figure ─────────────────────────────────────────────────────────────────
    if generate_figure:

        subj_sorted = overall_mean.sort_values(ascending=True).index.tolist()
        x_pos = {s: i for i, s in enumerate(subj_sorted)}

        def darken(color, factor=0.6):
            rgb = np.array(mcolors.to_rgb(color))
            return tuple(np.clip(rgb * factor, 0, 1))

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, subj in enumerate(subj_sorted):
            ax.bar(i, overall_mean[subj],
                   color=SUBJECT_COLORS[subj], edgecolor="white", width=0.6, zorder=2)

        for subj in subj_sorted:
            vals = trial_means[trial_means["subject"] == subj]["mean_speed"].values
            ax.scatter(np.full(len(vals), x_pos[subj]), vals,
                       color=darken(SUBJECT_COLORS[subj]), s=8, alpha=0.5, zorder=3)

        # ── CLD grouping lines ──────────────────────────────────────────────────
        y_top = max(overall_mean[s] for s in subj_sorted)
        y_range = trial_means["mean_speed"].max()
        base_h = y_top + y_range * 0.10
        step_h = y_range * 0.07
        tick_h = y_range * 0.015

        all_letters = sorted(set("".join(cld.values())))
        for li, letter in enumerate(all_letters):
            group = [s for s in subj_sorted if letter in cld[s]]
            xs = [x_pos[s] for s in group]
            x_left = min(xs) - 0.25
            x_right = max(xs) + 0.25
            h = base_h + li * step_h

            if len(group) == 1:
                ax.plot([xs[0], xs[0]], [h - tick_h, h],
                        color="black", linewidth=1.5, zorder=4)
            else:
                ax.plot([x_left, x_right], [h, h],
                        color="black", linewidth=1.5, zorder=4, solid_capstyle="round")
                for xend in [x_left, x_right]:
                    ax.plot([xend, xend], [h - tick_h, h],
                            color="black", linewidth=1.5, zorder=4)

            ax.text((x_left + x_right) / 2, h + tick_h * 0.5,
                    letter.upper(), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        top_line = base_h + (len(all_letters) - 1) * step_h + y_range * 0.08
        ax.set_ylim(0, top_line)

        ax.set_xticks(np.arange(len(subj_sorted)))
        ax.set_xticklabels([SUBJECT_ANON[s] for s in subj_sorted], fontsize=10)
        ax.set_ylabel("Mean speed (m/s)")
        ax.set_xlabel("")
        ax.set_title(f"Mean speed per subject — {joint}\n"
                     f"Lines group subjects not significantly different from each other (post-hoc, α=.05)")
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

        plt.tight_layout()
        save_figure(fig, PATH_FIGURES,
                    f"fig{str(n).zfill(2)}_mean_speed_by_subject_{joint.lower()}.png", **SAVE_KW)
        plt.close(fig)


def mean_speed_by_joint_by_subject(n, df, generate_figure=True):
    print_section_title(f"{n}. MEAN SPEED BY JOINT BY SUBJECT")

    mean_speed_subj = (df.groupby(["subject", "label"])["value"].mean().unstack("label").reindex(columns=joints))
    print("\nMean speed per joint per subject:")
    print(mean_speed_subj.round(4).to_csv(sep="\t"))

    if generate_figure:
        plot_dict_subj = {joint: mean_speed_subj[joint].values for joint in joints}
        n_subj = len(subjects)
        n_cols_grid = math.ceil(math.sqrt(len(subjects)))
        plot_silhouette(plot_dict_subj,
                        title="Mean speed per joint per subject",
                        title_silhouette=[s.split("_")[1] for s in subjects],
                        title_scale="Mean speed (m/s)",
                        min_scale=0, max_scale=float(mean_speed_subj.max().max()),
                        n_cols=n_cols_grid,
                        show=False,
                        color_scheme="celsius",
                        path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_mean_speed_by_joint_by_subject.png"),
                        resolution=(1000, 1080),
                        display_ratio=0.95,
                        verbosity=0)


def mean_speed_by_joint_by_subject_normalized(n, df, generate_figure=True):
    print_section_title(f"{n}. NORMALIZED MEAN SPEED BY SUBJECT")

    mean_speed_subj = (df.groupby(["subject", "label"])["value"].mean().unstack("label").reindex(columns=joints))
    subject_means = mean_speed_subj.replace(0, np.nan).mean(axis=1)
    normalized = mean_speed_subj.divide(subject_means, axis=0)

    print("\nNormalized mean speed per joint per subject (ratio to subject mean):")
    print(normalized.round(3).to_csv(sep="\t"))

    # Spearman correlation matrix between subject profiles
    print("\nSpearman correlation between subject normalized profiles:")
    if USE_REREF:
        norm_profiles = normalized.drop(columns=["SpineBase"], errors="ignore")
    else:
        norm_profiles = normalized.copy()
    corr_matrix = norm_profiles.T.corr(method="spearman")
    print(corr_matrix.round(3).to_csv(sep="\t"))

    if generate_figure:
        plot_dict_norm = {joint: normalized[joint].values for joint in joints}
        vmax = float(normalized.replace([np.inf, -np.inf], np.nan).max().max())
        plot_silhouette(plot_dict_norm,
                        title="Normalized mean speed per joint per subject",
                        title_silhouette=[s.split("_")[1] for s in subjects],
                        title_scale="Ratio to subject mean",
                        min_scale=0, max_scale=vmax,
                        n_cols=3,
                        color_scheme="celsius",
                        show=False,
                        path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_normalized_speed_by_subject.png"),
                        resolution=(1000, 1080),
                        verbosity=0)


def mean_speed_deviation_by_joint_by_subject(n, df, generate_figure=True):
    print_section_title(f"{n}. MEAN SPEED DEVIATION BY JOINT BY SUBJECT")

    mean_speed_subj = (df.groupby(["subject", "label"])["value"].mean().unstack("label").reindex(columns=joints))
    subject_means = mean_speed_subj.replace(0, np.nan).mean(axis=1)
    normalized = mean_speed_subj.divide(subject_means, axis=0)

    # ICC: consistency of joint recruitment pattern across subjects
    # Reshape to long: rows = joints, columns = subjects → ICC treats joints as targets, subjects as raters
    if USE_REREF:
        norm_clean = normalized.drop(columns=["SpineBase"], errors="ignore")
    else:
        norm_clean = normalized.copy()

    group_mean = norm_clean.mean(axis=0)
    deviation = norm_clean.subtract(group_mean, axis=1)

    print("\nMean speed deviation by joint by subject:")
    print(deviation.round(3).to_csv(sep="\t"))

    icc_long = norm_clean.reset_index().melt(id_vars="subject", var_name="joint", value_name="ratio")
    icc_result = pg.intraclass_corr(data=icc_long, targets="joint", raters="subject", ratings="ratio")
    print("\nICC (consistency of normalized recruitment pattern across subjects):")
    print(icc_result[["Type", "ICC", "CI95", "pval"]].to_string())

    if generate_figure:
        vmax = float(deviation.replace([np.inf, -np.inf], np.nan).abs().max().max())

        plot_dict_dev = {joint: deviation[joint].values
                         for joint in joints if joint in deviation.columns}

        plot_silhouette(plot_dict_dev,
                        title="Deviation from group mean normalized speed",
                        title_silhouette=[s.split("_")[-1] for s in subjects],
                        title_scale="Deviation from group mean",
                        min_scale=-vmax, max_scale=vmax,
                        color_scheme=["#2166ac", "#f7f7f7", "#d6604d"],
                        n_cols=3,
                        show=False,
                        path_save=op.join(PATH_FIGURES,
                                          f"fig{str(n).zfill(2)}_joint_mean_speed_deviation_from_group_mean.png"),
                        resolution=(1000, 1080),
                        verbosity=0)


def joint_rank(n, df, generate_figure=True):
    print_section_title(f"{n}. OVERALL JOINT RANK")

    joints_upper = [j for j in joints if j not in LOWER_BODY_JOINTS]

    mean_speed = df.groupby("label")["value"].mean().reindex(joints_upper)
    mean_speed_subj = (df.groupby(["subject", "label"])["value"]
                       .mean().unstack("label").reindex(columns=joints_upper))
    rank_per_subject = mean_speed_subj.rank(axis=1, ascending=False).astype(int)
    mean_rank = rank_per_subject.mean().reindex(joints_upper)

    print("\nOverall joint ranking by mean speed:")
    ranking = mean_speed.sort_values(ascending=False)
    for i, (joint, val) in enumerate(ranking.items()):
        print(f"  {i + 1:2d}. {joint:<20} {val:.4f}")

    print("\nMean rank per joint across subjects (lower = more dominant):")
    for joint, val in mean_rank.sort_values().items():
        print(f"  {joint:<20} {val:.2f}")

    # if USE_REREF:
    #     rank_matrix = rank_per_subject[[j for j in joints_upper if j != "SpineBase"]].values
    # else:
    rank_matrix = rank_per_subject.values
    n_ranked = rank_matrix.shape[1]
    W, chi2, p = kendalls_w(rank_matrix)
    interp = "excellent" if W > 0.9 else "good" if W > 0.7 else "moderate" if W > 0.5 else "poor"
    sig = Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p < 0.05 else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
    print(f"\nKendall's W (consistency of joint ranking across subjects):")
    print(f"  W = {W:.3f}, chi2({n_ranked - 1}) = {chi2:.3f}, p = {p:.4f} — {interp} {sig}")

    if generate_figure:
        n_joints = len(joints_upper)
        overall_rank = mean_speed.rank(ascending=False).astype(int)
        plot_silhouette({joint: int(overall_rank[joint]) for joint in joints_upper},
                        # title="Joint rank across subjects",
                        title_scale="Overall rank",
                        min_scale=1, max_scale=float(n_joints),
                        color_scheme="celsius_r",
                        show=False,
                        path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_overall_joint_rank.png"),
                        path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"),
                        show_values=True,
                        value_format=".0f",
                        resolution=(800, 1080),
                        show_scale=False,
                        verbosity=0)


def joint_rank_stability(n, df, generate_figure=True):
    print_section_title(f"{n}. OVERALL JOINT STABILITY")

    n_permutations = 10000

    mean_speed = df.groupby("label")["value"].mean().reindex(joints)
    mean_speed_subj = (df.groupby(["subject", "label"])["value"]
                       .mean().unstack("label").reindex(columns=joints))
    rank_per_subject = mean_speed_subj.rank(axis=1, ascending=False).astype(int)

    rank_sd = rank_per_subject.std()
    print("\nRank SD per joint (lower = more stable across subjects):")
    print(rank_sd.sort_values().round(2).to_string())

    if USE_REREF:
        joints_for_test = [j for j in joints if j != "SpineBase"]
    else:
        joints_for_test = joints.copy()

    rng = np.random.default_rng(42)
    n_subj, n_joints = rank_per_subject[joints_for_test].shape
    observed_sd = rank_per_subject[joints_for_test].std()

    # Build null distribution by permuting ranks within each subject
    null_sds = np.zeros((n_permutations, n_joints))
    data = rank_per_subject[joints_for_test].values
    for perm in range(n_permutations):
        permuted = np.apply_along_axis(rng.permutation, axis=1, arr=data)
        null_sds[perm] = permuted.std(axis=0)

    # p-value: proportion of null SDs >= observed SD (one-tailed: is this joint unstable?)
    p_values = np.array([(null_sds[:, j] >= observed_sd.iloc[j]).mean()
                         for j in range(n_joints)])

    # FDR correction
    reject, p_adj, _, _ = multipletests(p_values, method="fdr_bh")

    stability = pd.DataFrame({
        "joint": joints_for_test,
        "rank_sd": observed_sd.values,
        "p_raw": p_values,
        "p_adj": p_adj,
        "unstable": reject
    }).sort_values("rank_sd", ascending=False)

    print(f"\nRank stability per joint (permutation test, FDR-corrected, "
          f"n_permutations={n_permutations}):")
    print(f"  Unstable joints (p_adj < .05): "
          f"{stability[stability['unstable']]['joint'].tolist()}")
    print(stability.to_string(index=False))

    if generate_figure:
        n_joints = len(joints)
        plot_silhouette({joint: rank_sd[joint] for joint in joints},
                        title="Joint rank stability",
                        title_scale="Joint rank SD",
                        color_scheme="celsius",
                        show=False,
                        path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_joint_rank_stability.png"),
                        show_values=False,
                        resolution=(800, 1080),
                        show_scale=True,
                        verbosity=0)


def joint_rank_by_subject(n, df, generate_figure=True):
    print_section_title(f"{n}. JOINT RANK BY SUBJECT")

    mean_speed = df.groupby("label")["value"].mean().reindex(joints)
    mean_speed_subj = (df.groupby(["subject", "label"])["value"]
                       .mean().unstack("label").reindex(columns=joints))
    rank_per_subject = mean_speed_subj.rank(axis=1, ascending=False).astype(int)
    print(rank_per_subject.to_string())

    overall_rank = mean_speed.rank(ascending=False)
    print("\nSpearman correlation of each subject's rank vs. overall rank:")
    for subj in subjects:
        subj_rank = rank_per_subject.loc[subj]
        r, p = spearmanr(subj_rank, overall_rank)
        print(f"  {subj}: r = {r:.3f}, p = {p:.4f}")

    if generate_figure:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(rank_per_subject, ax=ax, cmap="YlOrRd_r", annot=True, fmt="d",
                    linewidths=0.5, cbar_kws={"label": "Rank (1 = most dominant)"})
        ax.set_title("Joint rank per subject (1 = fastest)")
        ax.set_xlabel("Joint")
        ax.set_ylabel("Subject")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_joint_rank_by_subject.png", **SAVE_KW)


def trial_order(n, df, joint=None, generate_figure=True, ax=None):
    print_section_title(f"{n}. TRIAL ORDER")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    # Filter to specific joint(s) if requested
    if joint is not None:
        joint_list = [joint] if isinstance(joint, str) else list(joint)
        df_test = df_test[df_test["label"].isin(joint_list)]
        joint_label = joint if isinstance(joint, str) else "+".join(joint_list)
    else:
        joint_label = "all joints"

    # Mean speed per trial (collapsed across joints and subjects)
    trial_means = (df_test.groupby(["subject", "trial"])["value"]
                   .mean().reset_index())
    trial_means.columns = ["subject", "trial", "mean_speed"]

    # Extract trial order (position within subject)
    trial_means["trial_order"] = (trial_means.groupby("subject")["trial"]
                                  .transform(lambda x: pd.factorize(x)[0] + 1))
    trial_means = trial_means.sort_values(["subject", "trial"])

    # Pearson is preferred here: session order is a metric integer variable and the
    # relationship with duration is expected to be linear.
    r_all, p_all = stats.pearsonr(trial_means["trial_order"], trial_means["mean_speed"])
    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_all < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"\nPearson correlation (session order vs retelling duration):")
    print(f"  r = {r_all:.3f}, p = {p_all:.4f} {sym}")

    # Group-level test: is there a consistent trend across subjects?
    # Use LMM with trial_order as fixed effect and subject as random effect
    print(f"\nLinear Mixed Model: mean_speed ~ trial_order + (1|subject)  [{joint_label}]")
    lmm = smf.mixedlm("mean_speed ~ trial_order", trial_means,
                       groups=trial_means["subject"])
    lmm_fit = lmm.fit(reml=True)
    beta = lmm_fit.params["trial_order"]
    lmm_p = lmm_fit.pvalues["trial_order"]
    ci = lmm_fit.conf_int().loc["trial_order"]
    sym = Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if lmm_p < 0.05 else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
    print(f"  β(trial_order) = {beta:.6f}, 95% CI [{ci[0]:.6f}, {ci[1]:.6f}], "
          f"p = {lmm_p:.4f} {sym}")
    direction = "increase" if beta > 0 else "decrease"
    print(f"  Interpretation: mean speed tends to {direction} by "
          f"{abs(beta)*1000:.4f} mm/s per trial ({'significant' if lmm_p < 0.05 else 'non-significant'})")

    if generate_figure:
        def fmt_nolead(value):
            s = f"{value:.2f}"
            if s.startswith("0."):
                return s[1:]
            elif s.startswith("-0."):
                return "-" + s[2:]
            return s

        def fmt_p(value):
            if value < 0.001:
                return "< .001"
            elif value < 0.01:
                return "< .01"
            return "= " + fmt_nolead(value)

        if p_all > 0.05:
            text_color = "#808080"
            fontweight = "normal"
        else:
            text_color = "black"
            fontweight = "bold"

        external_ax = ax is not None
        if not external_ax:
            fig, ax = plt.subplots(figsize=(8, 5))
        group_trend = trial_means.groupby("trial_order")["mean_speed"].agg(["mean", "sem"]) * 100
        ax.fill_between(group_trend.index,
                        group_trend["mean"] - group_trend["sem"],
                        group_trend["mean"] + group_trend["sem"],
                        alpha=0.2, color="#028090")
        ax.plot(group_trend.index, group_trend["mean"],
                color="#028090", alpha=0.3, linewidth=2)
        m_all, b_all, _, _, _ = stats.linregress(trial_means["trial_order"],
                                                  trial_means["mean_speed"])
        x_lmm = np.array([trial_means["trial_order"].min(),
                          trial_means["trial_order"].max()])
        # ax.plot(x_lmm, (lmm_fit.params["Intercept"] + beta * x_lmm) * 100,
        #         color="black", alpha=0.3, linewidth=1.5, linestyle="--",
        #         label=f"LMM: β = {beta:.5f}, p = {lmm_p:.3f}")
        ax.plot(x_lmm, (m_all * x_lmm + b_all) * 100, color="black", alpha=0.3, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Trial order")
        ax.set_ylabel("Mean speed (cm/s)")
        # ax.set_title(f"Group-level trial order trend — {joint_label} (mean ± SEM across subjects)")
        y_range = group_trend["mean"].max() - group_trend["mean"].min()
        y_ref = (m_all * x_lmm[-1] + b_all) * 100
        ax.text(x_lmm[-1], y_ref + y_range * 0.05, f"r = {fmt_nolead(r_all)}",
                ha="right", va="bottom", fontsize=12, fontweight=fontweight, color=text_color)
        ax.text(x_lmm[-1], y_ref - y_range * 0.05, f"p {fmt_p(p_all)}",
                ha="right", va="top", fontsize=12, fontweight=fontweight, color=text_color)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if not external_ax:
            fig.tight_layout()
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_trial_order_group_{joint_label}.png", **SAVE_KW)
            plt.close(fig)

def trial_order_by_subject(n, df, joint=None, generate_figure=True):
    print_section_title(f"{n}. TRIAL ORDER BY SUBJECT")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    # Filter to specific joint(s) if requested
    if joint is not None:
        joint_list = [joint] if isinstance(joint, str) else list(joint)
        df_test = df_test[df_test["label"].isin(joint_list)]
        joint_label = joint if isinstance(joint, str) else "+".join(joint_list)
    else:
        joint_label = "all joints"

    # Mean speed per trial (collapsed across joints and subjects)
    trial_means = (df_test.groupby(["subject", "trial"])["value"]
                   .mean().reset_index())
    trial_means.columns = ["subject", "trial", "mean_speed"]

    # Extract trial order (position within subject)
    trial_means["trial_order"] = (trial_means.groupby("subject")["trial"]
                                  .transform(lambda x: pd.factorize(x)[0] + 1))

    # ── Spearman correlation per subject ───────────────────────────────────────
    print("\nSpearman correlation between trial order and mean speed per subject:")
    rs, ps = [], []
    for subj in subjects:
        subj_data = trial_means[trial_means["subject"] == subj]
        r, p = spearmanr(subj_data["trial_order"], subj_data["mean_speed"])
        rs.append(r)
        ps.append(p)
        if p < 0.05:
            if r > 0:
                sym = Fore.BLUE + "\033[1m↗\033[0m" + Style.RESET_ALL
            else:
                sym = Fore.YELLOW + "\033[1m↘\033[0m" + Style.RESET_ALL
        else:
            sym = Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
        print(f"  {subj}: r = {r:.3f}, p = {p:.4f} {sym}")

    if generate_figure:
        n_subj = len(subjects)
        n_cols = 3
        n_rows = math.ceil(n_subj / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharey=True)
        axes = axes.flatten()
        for i, subj in enumerate(subjects):
            ax = axes[i]
            subj_data = trial_means[trial_means["subject"] == subj].sort_values("trial_order")
            color = SUBJECT_COLORS[subj]
            ax.scatter(subj_data["trial_order"], subj_data["mean_speed"],
                       color=color, alpha=0.5, s=25, zorder=3)
            m, b, _, _, _ = stats.linregress(subj_data["trial_order"],
                                             subj_data["mean_speed"])
            x_line = np.array([subj_data["trial_order"].min(),
                               subj_data["trial_order"].max()])
            r_s, p_s = spearmanr(subj_data["trial_order"], subj_data["mean_speed"])
            lw = 2.5 if p_s < 0.05 else 1.5
            ls = "-" if p_s < 0.05 else "--"
            ax.plot(x_line, m * x_line + b, color=color, lw=lw, linestyle=ls)
            sig = "✓" if p_s < 0.05 else "ns"
            ax.set_title(f"{subj.split('_')[-1]}  (r = {r_s:.2f}, {sig})", fontsize=11)
            ax.set_xlabel("Trial order")
            ax.set_ylabel("Mean speed (m/s)" if i % n_cols == 0 else "")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Trial order vs mean speed per subject — {joint_label} (solid = p < .05)",
                     fontsize=13, y=1.01)
        fig.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_trial_order_per_subject_{joint_label}.png", **SAVE_KW)
        plt.close(fig)

def mean_speed_by_cartoon(n, df, joint=None, generate_figure=True):
    print_section_title(f"{n}. MEAN SPEED BY CARTOON")

    import pingouin as pg

    min_eta2 = pg.power_anova(eta_squared=None, k=81, n=4, power=0.80, alpha=0.05)
    min_f = np.sqrt(min_eta2 / (1 - min_eta2))
    print(f"Minimum detectable η²: {min_eta2:.4f}")
    print(f"Equivalent Cohen's f:  {min_f:.4f}")

    # Load show mapping from the info Excel file
    df_info = pd.read_excel(PATH_INFO, "Videos (excl.)")
    cartoon_to_show = df_info.drop_duplicates("Cartoon").set_index("Cartoon")["Show"]

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    if joint is not None:
        joint_list = [joint] if isinstance(joint, str) else list(joint)
        df_test = df_test[df_test["label"].isin(joint_list)]
        joint_label = joint if isinstance(joint, str) else "+".join(joint_list)
    else:
        joint_label = "all joints"

    # Mean speed per trial collapsed across joints
    trial_means = (df_test.groupby(["subject", "trial"])["value"].mean().reset_index().rename(columns={"value": "mean_speed", "trial": "cartoon"}))
    trial_means["show"] = trial_means["cartoon"].map(cartoon_to_show)

    by_cartoon = (trial_means.groupby("cartoon")["mean_speed"].agg(["mean", "std", "min", "max", "count"]))
    by_cartoon.columns = ["Mean (m/s)", "SD (m/s)", "Min (m/s)", "Max (m/s)", "N trials"]
    by_cartoon = by_cartoon.round(4)

    # print(f"Descriptive statistics per cartoon (N = {len(by_cartoon)} cartoons) [{joint_label}]:")
    # print(by_cartoon.to_string())

    print(f"\nSummary across cartoons:")
    print(f"  Slowest mean speed : {by_cartoon['Mean (m/s)'].min():.4f} m/s"
          f"  ({by_cartoon['Mean (m/s)'].idxmin()})")
    print(f"  Fastest mean speed : {by_cartoon['Mean (m/s)'].max():.4f} m/s"
          f"  ({by_cartoon['Mean (m/s)'].idxmax()})")
    print(f"  Grand mean         : {by_cartoon['Mean (m/s)'].mean():.4f} m/s"
          f"  (SD = {by_cartoon['Mean (m/s)'].std():.4f} m/s)")
    print(f"  Average N per cartoon : {by_cartoon['N trials'].mean():.1f}"
          f"  (range: {int(by_cartoon['N trials'].min())}–{int(by_cartoon['N trials'].max())})")

    cartoons = sorted(trial_means["cartoon"].unique())

    grand_mean = trial_means.groupby("cartoon")["mean_speed"].mean().mean()
    cartoon_means = trial_means.groupby("cartoon")["mean_speed"].mean()
    ref_cartoon = (cartoon_means - grand_mean).abs().idxmin()
    print(f"\nReference cartoon (closest to grand mean): {ref_cartoon} "
          f"({cartoon_means[ref_cartoon]:.4f} m/s, grand mean = {grand_mean:.4f} m/s)")

    # LMM: mean_speed ~ cartoon + (1|subject)
    # Cartoon is categorical so use C() notation
    print("\nLinear Mixed Model: mean_speed ~ cartoon + (1|subject)")
    lmm = smf.mixedlm(f"mean_speed ~ C(cartoon, Treatment('{ref_cartoon}'))",
                       trial_means, groups=trial_means["subject"])
    lmm_fit = lmm.fit(reml=True)
    # print(lmm_fit.summary())

    # Extract p-values for each cartoon level
    # print("\nCartoon fixed effects:")
    # for param, beta, p in zip(lmm_fit.params.index,
    #                           lmm_fit.params.values,
    #                           lmm_fit.pvalues.values):
    #     if "cartoon" in param.lower():
    #         sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p < 0.05
    #                else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    #         print(f"  {param}: β = {beta:.4f}, p = {p:.4f} {sym}")

    # Aggregate to one mean per subject per cartoon before testing
    subj_cartoon_means = (trial_means.groupby(["subject", "cartoon"])["mean_speed"]
                          .mean().reset_index())

    groups = [grp["mean_speed"].values for _, grp in subj_cartoon_means.groupby("cartoon")]
    k = len(groups)
    n_total = len(subj_cartoon_means)

    h_stat, p_kruskal = stats.kruskal(*groups)
    eta2 = h_stat / (n_total - 1)

    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_kruskal < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"\nKruskal-Wallis (one mean per subject per cartoon):")
    print(f"  H({k - 1}) = {h_stat:.3f}, p = {p_kruskal:.4f}  {sym}")
    print(f"  Effect size: η² = {eta2:.3f}")

    if p_kruskal < 0.05:
        import scikit_posthocs as sp
        dunn = sp.posthoc_dunn(subj_cartoon_means, val_col="mean_speed",
                               group_col="cartoon", p_adjust="holm")
        print("\nPost-hoc: Dunn's test (Holm correction)")
        print(dunn.to_string())
    else:
        print("\nNo significant difference in mean speed across cartoons.")

    if generate_figure:
        cartoon_show = trial_means.groupby("cartoon")["show"].first()
        plot_df = (trial_means.groupby("cartoon")["mean_speed"]
                   .mean().to_frame().join(cartoon_show))
        plot_df = plot_df.sort_values("mean_speed")
        cartoon_to_y = {c: i for i, c in enumerate(plot_df.index)}

        fig, ax = plt.subplots(figsize=(10, max(8, len(plot_df) * 0.4)))
        y_pos = np.arange(len(plot_df))
        bar_colors = [SHOW_COLORS[s] for s in plot_df["show"]]

        ax.barh(y_pos, plot_df["mean_speed"],
                color=bar_colors, edgecolor="white", height=0.75)

        # Individual subject dots per cartoon
        for subj, grp in trial_means.groupby("subject"):
            subj_cartoon_means = grp.groupby("cartoon")["mean_speed"].mean()
            ys = [cartoon_to_y[c] for c in subj_cartoon_means.index]
            ax.scatter(subj_cartoon_means.values, ys,
                       color=SUBJECT_COLORS[subj], s=18, zorder=3,
                       alpha=0.85, linewidths=0.3, edgecolors="white")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df.index, fontsize=7)
        ax.set_xlabel("Mean speed (m/s)")
        ax.set_title(f"Mean movement speed per cartoon — {joint_label} (N = {len(plot_df)}, sorted by mean)")

        from matplotlib.patches import Patch
        import matplotlib.lines as mlines
        show_handles = [Patch(facecolor=SHOW_COLORS[s], label=s) for s in SHOW_COLORS]
        subject_handles = [
            mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor=SUBJECT_COLORS[s],
                          markersize=6, label=s.split("_")[-1])
            for s in sorted(SUBJECT_COLORS)
        ]
        ax.legend(handles=show_handles + subject_handles,
                  title="Show / Subject", fontsize=9, title_fontsize=9,
                  loc="lower right")

        fig.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_mean_speed_by_cartoon_{joint_label}.png", **SAVE_KW)
        plt.close(fig)


def mean_speed_by_show(n, df, joint=None, generate_figure=True):
    print_section_title(f"{n}. MEAN SPEED BY SHOW")

    # Load show mapping from the info Excel file
    df_info = pd.read_excel(PATH_INFO, "Videos (excl.)")
    cartoon_to_show = df_info.drop_duplicates("Cartoon").set_index("Cartoon")["Show"]

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    # Filter to specific joint(s) if requested
    if joint is not None:
        joint_list = [joint] if isinstance(joint, str) else list(joint)
        df_test = df_test[df_test["label"].isin(joint_list)]
        joint_label = joint if isinstance(joint, str) else "+".join(joint_list)
    else:
        joint_label = "all joints"

    # Mean speed per trial collapsed across joints
    trial_means = (df_test.groupby(["subject", "trial"])["value"]
                   .mean().reset_index()
                   .rename(columns={"value": "mean_speed", "trial": "cartoon"}))
    trial_means["show"] = trial_means["cartoon"].map(cartoon_to_show)

    # ── Descriptive statistics ─────────────────────────────────────────────────
    by_show = (trial_means.groupby("show")["mean_speed"]
               .agg(["mean", "std", "min", "max", "count"]))
    by_show.columns = ["Mean (m/s)", "SD (m/s)", "Min (m/s)", "Max (m/s)", "N trials"]
    by_show = by_show.round(4)

    print(f"Descriptive statistics per show (N = {len(by_show)} shows) [{joint_label}]:")
    print(by_show.to_string())

    # ── Friedman test (omnibus) ────────────────────────────────────────────────
    shows_ordered = sorted(trial_means["show"].unique())
    subj_show_means = (trial_means.groupby(["subject", "show"])["mean_speed"]
                       .mean().unstack("show"))
    chi2_f, p_f = stats.friedmanchisquare(*[subj_show_means[s].values
                                            for s in shows_ordered])
    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_f < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"\nFriedman test: χ²({len(shows_ordered) - 1}) = {chi2_f:.3f}, "
          f"p = {p_f:.4f} {sym}")

    # ── Pairwise Wilcoxon post-hoc (if significant) ────────────────────────────
    pval_lookup = {}
    if p_f < 0.05:
        print("\nPost-hoc: Pairwise Wilcoxon signed-rank tests (Holm correction)")
        pairs = list(combinations(shows_ordered, 2))
        pair_ps = []
        for s1, s2 in pairs:
            stat, p = stats.wilcoxon(subj_show_means[s1], subj_show_means[s2])
            pair_ps.append(p)

        reject, p_adj, _, _ = multipletests(pair_ps, method="holm")
        for (s1, s2), p_raw, p_adj_, sig in zip(pairs, pair_ps, p_adj, reject):
            pval_lookup[(s1, s2)] = p_adj_
            sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if sig
                   else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
            print(f"  {s1} vs {s2}: p_raw = {p_raw:.4f}, "
                  f"p_adj = {p_adj_:.4f} {sym}")
    else:
        print("\nNo significant difference in mean speed across shows.")

    if generate_figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=trial_means, x="show", y="mean_speed",
                    order=shows_ordered, hue="show", palette=SHOW_COLORS,
                    width=0.5, fliersize=0, legend=False, ax=ax)
        sns.stripplot(data=trial_means, x="show", y="mean_speed",
                      order=shows_ordered, hue="show", palette=SHOW_COLORS,
                      alpha=0.3, jitter=True, size=3, legend=False, ax=ax)

        # Significance brackets
        if pval_lookup:
            y_max = trial_means["mean_speed"].max()
            y_range = y_max - trial_means["mean_speed"].min()
            bracket_base = y_max + y_range * 0.05
            bracket_step = y_range * 0.08
            tick = y_range * 0.015

            def get_stars(p):
                if p < 0.001: return "***"
                elif p < 0.01: return "**"
                elif p < 0.05: return "*"
                else: return None

            bracket_n = 0
            for s1, s2 in combinations(shows_ordered, 2):
                p = pval_lookup.get((s1, s2)) or pval_lookup.get((s2, s1))
                if p is None or p >= 0.05:
                    continue
                stars = get_stars(p)
                x1 = shows_ordered.index(s1)
                x2 = shows_ordered.index(s2)
                h = bracket_base + bracket_n * bracket_step
                ax.plot([x1, x1, x2, x2], [h, h + tick, h + tick, h],
                        color="black", linewidth=1.2)
                ax.text((x1 + x2) / 2, h + tick, stars,
                        ha="center", va="bottom", fontsize=11)
                bracket_n += 1

            ax.set_ylim(top=bracket_base + bracket_n * bracket_step + y_range * 0.1)

        sig_str = "p < .05" if p_f < 0.05 else "ns"
        ax.set_xlabel("Show")
        ax.set_ylabel("Mean speed (m/s)")
        ax.set_title(f"Mean movement speed by show — {joint_label} ({sig_str})")

        fig.tight_layout()
        save_figure(fig, PATH_FIGURES,
                    f"fig{str(n).zfill(2)}_mean_speed_by_show_{joint_label}.png",
                    **SAVE_KW)
        plt.close(fig)


def mean_speed_by_condition(n, df, joint=None, generate_figure=True, ax=None, p_for_plot=None, p_label_override=None):
    print_section_title(f"{n}. MEAN SPEED BY CONDITION")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    # Filter to specific joint(s) if requested
    if joint is not None:
        joint_list = [joint] if isinstance(joint, str) else list(joint)
        df_test = df_test[df_test["label"].isin(joint_list)]
        joint_label = joint if isinstance(joint, str) else "+".join(joint_list)
    else:
        joint_label = "all joints"

    # Mean speed per trial collapsed across joints
    trial_means = (df_test.groupby(["subject", "trial", "condition"])["value"]
                   .mean().reset_index()
                   .rename(columns={"value": "mean_speed"}))

    conditions = sorted(trial_means["condition"].unique())

    # ── Descriptive statistics ─────────────────────────────────────────────────
    by_condition = (trial_means.groupby("condition")["mean_speed"]
                    .agg(["mean", "std", "min", "max", "count"]))
    by_condition.columns = ["Mean (m/s)", "SD (m/s)", "Min (m/s)", "Max (m/s)", "N trials"]
    by_condition = by_condition.round(4)

    print(f"Descriptive statistics per condition [{joint_label}]:")
    print(by_condition.to_string())

    print(f"\nSummary:")
    for cond in conditions:
        print(f"  {cond}: M = {by_condition.loc[cond, 'Mean (m/s)']:.4f} m/s, "
              f"SD = {by_condition.loc[cond, 'SD (m/s)']:.4f} m/s")

    # ── Stats ──────────────────────────────────────────────────────────────────
    # Per-subject means per condition
    subj_cond_means = (trial_means.groupby(["subject", "condition"])["mean_speed"]
                       .mean().unstack("condition"))

    if len(conditions) == 2:
        c1, c2 = conditions
        stat, p = stats.wilcoxon(subj_cond_means[c1], subj_cond_means[c2])
        sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p < 0.05
               else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
        print(f"\nWilcoxon signed-rank test ({c1} vs {c2}) [{joint_label}]:")
        print(f"  W = {stat:.3f}, p = {p:.4f} {sym}")
        direction = "faster" if by_condition.loc[c1, "Mean (m/s)"] > by_condition.loc[c2, "Mean (m/s)"] else "slower"
        print(f"  Interpretation: {c1} is {direction} than {c2} "
              f"({'significant' if p < 0.05 else 'non-significant'})")
        pval = p

        # Normality of difference scores
        diff_scores = subj_cond_means[c1].values - subj_cond_means[c2].values
        w_norm, p_norm = stats.shapiro(diff_scores)
        print(f"\nShapiro-Wilk on difference scores: W = {w_norm:.3f}, p = {p_norm:.3f}")

        # Paired t-test
        t_stat, p_t = stats.ttest_rel(subj_cond_means[c1], subj_cond_means[c2])
        d = diff_scores.mean() / diff_scores.std(ddof=1)
        print(f"Paired t-test: t({len(diff_scores) - 1}) = {t_stat:.3f}, p = {p_t:.3f}, d = {d:.3f}")

        # Power
        from statsmodels.stats.power import TTestPower
        power = TTestPower().power(effect_size=abs(d), nobs=len(diff_scores),
                                   alpha=0.05, alternative="two-sided")
        print(f"Power: 1-β = {power:.2f}")
    else:
        # Friedman for more than 2 conditions
        chi2_f, p_f = stats.friedmanchisquare(*[subj_cond_means[c].values
                                                 for c in conditions])
        sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_f < 0.05
               else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
        print(f"\nFriedman test: χ²({len(conditions) - 1}) = {chi2_f:.3f}, "
              f"p = {p_f:.4f} {sym}")
        pval = p_f

        if p_f < 0.05:
            print("\nPost-hoc: Pairwise Wilcoxon signed-rank tests (Holm correction)")
            pairs = list(combinations(conditions, 2))
            pair_ps = [stats.wilcoxon(subj_cond_means[c1],
                                      subj_cond_means[c2])[1]
                       for c1, c2 in pairs]
            reject, p_adj, _, _ = multipletests(pair_ps, method="holm")
            for (c1, c2), p_raw, p_adj_, sig in zip(pairs, pair_ps, p_adj, reject):
                sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if sig
                       else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
                print(f"  {c1} vs {c2}: p_raw = {p_raw:.4f}, "
                      f"p_adj = {p_adj_:.4f} {sym}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    if generate_figure and len(conditions) != 2:
        print("  Figure not implemented for more than 2 conditions.")

    elif generate_figure:
        from scipy.stats import t as t_dist
        from matplotlib.lines import Line2D

        # Which condition each subject saw first (Block 1)
        df_info = pd.read_excel(PATH_INFO, "Videos (excl.)")
        condition_block = (df_info.groupby(["Subject", "Masked?"])["Block"]
                           .first().unstack("Masked?"))
        condition_block.index = condition_block.index.str.lower()

        cond_first = {subj: condition_block.loc[subj.lower(), c1] == 1
                      for subj in subj_cond_means.index if subj.lower() in condition_block.index}
        for subj in subj_cond_means.index:
            if subj not in cond_first:
                print(f"  ⚠ {subj} not found in the info file — drawn as if {c1} was presented first.")
                cond_first[subj] = True

        external_ax = ax is not None
        if not external_ax:
            fig, ax = plt.subplots(figsize=(6, 6))

        means_cm = subj_cond_means[[c1, c2]].astype(float) * 100  # m/s → cm/s
        x = np.arange(len(conditions))
        offset_mean = 0.12  # Horizontal distance between the subject dots and the group mean diamond

        # Paired subject lines and dots: solid if c1 was presented first, dashed otherwise
        for subj in sorted(means_cm.index):
            row = means_cm.loc[subj]
            if row.isna().any():
                continue
            linestyle = "-" if cond_first[subj] else "--"
            ax.plot(x, [row[c1], row[c2]], color="#a0a0a0", lw=1.2, ls=linestyle, zorder=2)
            ax.scatter(x, [row[c1], row[c2]], color="#60606080", edgecolors="#30303080", s=50, zorder=3)

        # Group mean ± 95% CI, next to the dots, on the outer side
        ci_tops, x_means = [], []
        for i, c in enumerate([c1, c2]):
            color = CONDITION_COLORS[c]
            vals = means_cm[c].dropna().values
            m, sem = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
            ci = sem * t_dist.ppf(0.975, len(vals) - 1)
            ci_tops.append(m + ci)
            x_pos = x[i] - offset_mean if i == 0 else x[i] + offset_mean
            x_means.append(x_pos)
            ax.errorbar(x_pos, m, yerr=ci, fmt="D", color=darken(color), mfc=color, mec=darken(color), ms=8,
                        elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

        # Legend for the presentation order
        handles = [Line2D([0], [0], color="#a0a0a0", lw=1.2, ls="-", label=f"{c1.capitalize()} first"),
                   Line2D([0], [0], color="#a0a0a0", lw=1.2, ls="--", label=f"{c2.capitalize()} first")]
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2)

        ax.set_xlim(x[0] - 0.5, x[1] + 0.5)

        # Significance bracket (placed above both the data and the CIs)
        p_plot = p_for_plot if p_for_plot is not None else pval
        y_max = max(means_cm.max().max(), max(ci_tops)) * 1.01
        ax.plot([x_means[0], x_means[0], x_means[1], x_means[1]],
                [y_max, y_max * 1.005, y_max * 1.005, y_max],
                color="#808080", lw=1.1)
        if p_label_override is not None:
            p_label = p_label_override
        else:
            p_label = "n.s." if p_plot >= 0.05 else "***" if p_plot < 0.001 else "**" if p_plot < 0.01 else "*"
        ax.text(np.mean(x), y_max * 1.01, p_label, ha="center", va="bottom", fontsize=11)
        ax.set_ylim(top=y_max * 1.04)  # Leave room for the significance label inside the axes

        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in conditions])
        ylabel = "Mean speed (cm/s)" if joint is None else f"Mean speed (cm/s) — {joint_label}"
        ax.set_ylabel(ylabel, fontsize=11)

        if not external_ax:
            fig.tight_layout()
            save_figure(fig, PATH_FIGURES,
                        f"fig{str(n).zfill(2)}_mean_speed_by_condition_{joint_label}.png",
                        **SAVE_KW)
            plt.close(fig)


def mean_speed_by_condition_by_subject(n, df, joint=None, generate_figure=True):
    print_section_title(f"{n}. MEAN SPEED BY CONDITION BY SUBJECT")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    if joint is not None:
        joint_list = [joint] if isinstance(joint, str) else list(joint)
        df_test = df_test[df_test["label"].isin(joint_list)]
        joint_label = joint if isinstance(joint, str) else "+".join(joint_list)
    else:
        joint_label = "all joints"

    trial_means = (df_test.groupby(["subject", "trial", "condition"])["value"]
                   .mean().reset_index()
                   .rename(columns={"value": "mean_speed"}))

    conditions = sorted(trial_means["condition"].unique(), reverse=True)
    subj_cond_means = (trial_means.groupby(["subject", "condition"])["mean_speed"]
                       .mean().unstack("condition"))

    # ── Per-subject Wilcoxon ───────────────────────────────────────────────────
    print(f"\nPer-subject mean speed by condition [{joint_label}]:")
    print(subj_cond_means.round(4).to_string())

    print(f"\nMann-Whitney U test per subject [{joint_label}]:")
    c1, c2 = conditions
    for subj in subjects:
        masked_trials = trial_means[(trial_means["subject"] == subj) &
                                    (trial_means["condition"] == c1)]["mean_speed"]
        unmasked_trials = trial_means[(trial_means["subject"] == subj) &
                                      (trial_means["condition"] == c2)]["mean_speed"]
        # Use Mann-Whitney since trials are not paired within subject
        stat, p = stats.mannwhitneyu(masked_trials, unmasked_trials,
                                     alternative="two-sided")
        direction = "↘" if subj_cond_means.loc[subj, c1] > subj_cond_means.loc[subj, c2] else "↗"
        if p < 0.05:
            if direction == "↗":
                sym = (Fore.BLUE + f"\033[1m{direction}\033[0m" + Style.RESET_ALL)
            else:
                sym = (Fore.YELLOW + f"\033[1m{direction}\033[0m" + Style.RESET_ALL)
        else:
            sym = Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
        print(f"  {subj}: {c1} = {subj_cond_means.loc[subj, c1]:.4f}, "
              f"{c2} = {subj_cond_means.loc[subj, c2]:.4f}, "
              f"U = {stat:.1f}, p = {p:.4f} {sym}")

    if generate_figure:
        fig, ax = plt.subplots(figsize=(10, 5))

        df_plot = trial_means.copy()
        subject_labels = {s: s.split("_")[-1] for s in subjects}
        df_plot["SubjectShort"] = df_plot["subject"].map(subject_labels)
        order_short = [s.split("_")[-1] for s in sorted(subjects)]

        sns.boxplot(data=df_plot, x="SubjectShort", y="mean_speed", hue="condition",
                    order=order_short, hue_order=conditions,
                    palette=CONDITION_COLORS, width=0.6, fliersize=0, ax=ax)
        sns.stripplot(data=df_plot, x="SubjectShort", y="mean_speed", hue="condition",
                      order=order_short, hue_order=conditions,
                      palette=CONDITION_COLORS, alpha=0.3, jitter=True, size=3,
                      dodge=True, ax=ax, legend=False)

        # Significance markers
        subject_max = df_plot.groupby("SubjectShort")["mean_speed"].max()
        y_offset = trial_means["mean_speed"].max() * 0.03
        for i, subj_short in enumerate(order_short):
            subj_full = [s for s in subjects if s.split("_")[-1] == subj_short][0]
            masked_trials = trial_means[(trial_means["subject"] == subj_full) &
                                        (trial_means["condition"] == c1)]["mean_speed"]
            unmasked_trials = trial_means[(trial_means["subject"] == subj_full) &
                                          (trial_means["condition"] == c2)]["mean_speed"]
            _, p_subj = stats.mannwhitneyu(masked_trials, unmasked_trials,
                                           alternative="two-sided")
            if p_subj < 0.05:
                y_top = subject_max[subj_short] + y_offset
                stars = "***" if p_subj < 0.001 else "**" if p_subj < 0.01 else "*"
                ax.text(i, y_top, stars, ha="center", va="bottom",
                        fontsize=14, fontweight="bold")

        ax.set_xlabel("Subject")
        ax.set_ylabel("Mean speed (m/s)")
        ax.set_title(f"Subject × Condition interaction — {joint_label} (* p < .05)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(conditions)], labels[:len(conditions)], title="Condition")

        fig.tight_layout()
        save_figure(fig, PATH_FIGURES,
                    f"fig{str(n).zfill(2)}_mean_speed_by_condition_by_subject_{joint_label}.png",
                    **SAVE_KW)
        plt.close(fig)


def mean_speed_difference_by_condition(n, df, generate_figure=True):
    print_section_title(f"{n}. MEAN SPEED DIFFERENCE BY CONDITION")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    joints_test = [j for j in joints if j != "SpineBase"] if USE_REREF else joints.copy()

    c_masked, c_unmasked = "Masked", "Unmasked"

    # ── Per-subject mean speed: (subject × joint) for each condition ──────────
    subj_joint_cond = (
        df_test
        .groupby(["subject", "label", "condition"])["value"]
        .mean()
        .unstack("condition")
    )

    # ── Per-joint Wilcoxon + rank-biserial r ──────────────────────────────────
    results = []
    for joint in joints_test:
        if joint not in subj_joint_cond.index.get_level_values("label"):
            continue

        paired = (
            subj_joint_cond
            .xs(joint, level="label")[[c_masked, c_unmasked]]
            .dropna()
        )
        if len(paired) < 3:
            continue

        masked_vals = paired[c_masked].values
        unmasked_vals = paired[c_unmasked].values
        diffs = masked_vals - unmasked_vals
        n_pairs = len(paired)

        try:
            w_stat, p_raw = stats.wilcoxon(diffs, alternative="two-sided")
        except ValueError:
            # All differences are zero — no test possible
            w_stat, p_raw = 0.0, 1.0

        # Rank-biserial r: r = 1 - (2 * W_min) / (N*(N+1)/2)
        # scipy returns the minimum of W+ and W- so the formula gives the magnitude;
        # apply sign from mean difference direction.
        total_ranks = n_pairs * (n_pairs + 1) / 2
        r_abs = 1.0 - (2.0 * w_stat) / total_ranks
        r_rb = r_abs * np.sign(np.mean(diffs))

        results.append({
            "joint": joint,
            "mean_masked": masked_vals.mean(),
            "mean_unmasked": unmasked_vals.mean(),
            "mean_diff": diffs.mean(),
            "W": w_stat,
            "p_raw": p_raw,
            "r_rb": r_rb,
            "n_pairs": n_pairs,
        })

    df_res = pd.DataFrame(results)

    # ── Multiple-comparison correction ────────────────────────────────────────
    reject_fdr, p_adj, _, _ = multipletests(df_res["p_raw"], method="fdr_bh")
    df_res["p_adj"] = p_adj
    df_res["sig_fdr"] = reject_fdr
    df_res["sig_uncorr"] = df_res["p_raw"] < 0.05

    def _star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\nPer-joint Wilcoxon signed-rank: {c_masked} vs {c_unmasked}")
    print(f"Positive r_rb = {c_masked} faster  |  Negative r_rb = {c_unmasked} faster")
    header = f"  {'Joint':<22} {'M_mask':>8} {'M_unmask':>9} {'Δ':>8} {'r_rb':>6} {'p_raw':>8} {'p_adj':>8}  sig"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for _, row in df_res.sort_values("r_rb", ascending=False).iterrows():
        if row["sig_fdr"]:
            sig_str = Fore.GREEN + f"✓ FDR ({_star(row['p_raw'])})" + Style.RESET_ALL
        elif row["sig_uncorr"]:
            sig_str = Fore.YELLOW + f"✓ unc ({_star(row['p_raw'])})" + Style.RESET_ALL
        else:
            sig_str = Fore.RED + "✗" + Style.RESET_ALL

        print(f"  {row['joint']:<22} {row['mean_masked']:>8.4f} {row['mean_unmasked']:>9.4f} "
              f"{row['mean_diff']:>8.4f} {row['r_rb']:>6.3f} {row['p_raw']:>8.4f} "
              f"{row['p_adj']:>8.4f}  {sig_str}")

    n_sig_fdr = df_res["sig_fdr"].sum()
    n_sig_unc = df_res["sig_uncorr"].sum()
    n_max_subj = int(df_res["n_pairs"].max())
    print(f"\n  Significant (FDR-BH):          {n_sig_fdr}/{len(df_res)}")
    print(f"  Significant (uncorrected p<.05): {n_sig_unc}/{len(df_res)}")
    print(f"\n  ⚠ N={n_max_subj}: minimum achievable two-tailed p ≈ 0.031. "
          f"FDR correction is reported but likely underpowered. "
          f"Treat r_rb as the primary interpretive quantity.")

    if generate_figure:
        r_dict = dict(zip(df_res["joint"], df_res["r_rb"]))

        # SpineBase gets r = 0 (re-reference joint, excluded from test)
        if USE_REREF:
            r_dict["SpineBase"] = 0.0

        # Fill any joint not in test results
        for j in joints:
            r_dict.setdefault(j, 0.0)

        r_max = max(abs(v) for v in r_dict.values())
        r_max = max(r_max, 0.15)  # floor to avoid collapsed scale

        plot_silhouette(
            r_dict,
            title=(f"Condition effect on speed per joint\n"
                   f"rank-biserial r  (+1 = {c_masked} always faster)"),
            title_scale="Rank-biserial r",
            min_scale=-r_max,
            max_scale=r_max,
            color_scheme=["#2166ac", "#f7f7f7", "#d6604d"],  # blue → white → red
            font_size=0.75,
            show_values=True,
            value_format=".2f",
            show=False,
            path_save=op.join(PATH_FIGURES,
                              f"fig{str(n).zfill(2)}_condition_effect_per_joint_silhouette.png"),
            resolution=(800, 1080),
            verbosity=0,
        )


def bilateral_symmetry(n, df, generate_figure=True):
    print_section_title(f"{n}. BILATERAL SYMMETRY — LATERAL DOMINANCE")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    def _star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    # ── Per-subject mean speed per joint (collapsed across trials/conditions) ─
    subj_joint_mean = (
        df_test
        .groupby(["subject", "label"])["value"]
        .mean()
        .unstack("label")
    )

    # ── Compute AI per subject for each bilateral pair ────────────────────────
    results = []
    ai_matrix = {}  # pair_label -> array of per-subject AIs (for figure)

    for right_j, left_j in BILATERAL_PAIRS:
        # Skip pairs where either joint is missing from the dataframe
        if right_j not in subj_joint_mean.columns or left_j not in subj_joint_mean.columns:
            continue

        paired = subj_joint_mean[[right_j, left_j]].dropna()
        if len(paired) < 3:
            continue

        r_vals = paired[right_j].values
        l_vals = paired[left_j].values

        denom = r_vals + l_vals
        # Guard against zero denominator (both joints stationary)
        with np.errstate(invalid="ignore"):
            ai_vals = np.where(denom > 0, (r_vals - l_vals) / denom, 0.0)

        n_pairs = len(ai_vals)
        mean_ai = ai_vals.mean()
        pair_label = f"{right_j.replace('Right', '')} R/L"

        # One-sample Wilcoxon signed-rank against 0
        try:
            w_stat, p_raw = stats.wilcoxon(ai_vals, alternative="two-sided")
        except ValueError:
            # All AI values identical (e.g. all zero)
            w_stat, p_raw = 0.0, 1.0

        # Rank-biserial r (magnitude from W_min, sign from mean AI)
        total_ranks = n_pairs * (n_pairs + 1) / 2
        r_abs = 1.0 - (2.0 * w_stat) / total_ranks
        r_rb = r_abs * np.sign(mean_ai) if mean_ai != 0 else 0.0

        results.append({
            "pair": pair_label,
            "right_j": right_j,
            "left_j": left_j,
            "mean_ai": mean_ai,
            "sd_ai": ai_vals.std(),
            "W": w_stat,
            "p_raw": p_raw,
            "r_rb": r_rb,
            "n_subjects": n_pairs,
        })
        ai_matrix[pair_label] = {"right": right_j, "left": left_j, "ai": ai_vals}

    df_res = pd.DataFrame(results)

    # ── FDR correction ────────────────────────────────────────────────────────
    reject_fdr, p_adj, _, _ = multipletests(df_res["p_raw"], method="fdr_bh")
    df_res["p_adj"] = p_adj
    df_res["sig_fdr"] = reject_fdr
    df_res["sig_uncorr"] = df_res["p_raw"] < 0.05

    # ── Per-subject laterality test (trial-level paired Wilcoxon) ─────────────
    TOP_PAIRS = ["Hand", "Wrist", "Elbow", "Shoulder"]
    top_bilateral = [(r, l) for r, l in BILATERAL_PAIRS
                     if r.replace("Right", "") in TOP_PAIRS]

    print("\nPer-subject laterality (paired Wilcoxon across trials, right vs left):")
    print("Positive AI = right-dominant  |  Negative = left-dominant")

    for right_j, left_j in top_bilateral:
        pair_label = right_j.replace("Right", "")
        print(f"\n  {pair_label}:")
        header = f"    {'Subject':<12} {'AI':>7} {'W':>7} {'p':>8} {'p_FDR':>8}  sig"
        print(header)
        print("    " + "─" * (len(header) - 4))

        subj_results = []
        for subj, subj_df in df_test.groupby("subject"):
            r_trials = (subj_df[subj_df["label"] == right_j]
                        .groupby("trial")["value"].mean())
            l_trials = (subj_df[subj_df["label"] == left_j]
                        .groupby("trial")["value"].mean())
            paired = pd.concat([r_trials, l_trials], axis=1, keys=["R", "L"]).dropna()
            if len(paired) < 3:
                subj_results.append((subj, None, None, None, None))
                continue

            r_vals = paired["R"].values
            l_vals = paired["L"].values
            diffs = r_vals - l_vals
            denom = r_vals + l_vals
            with np.errstate(invalid="ignore"):
                ai_vals_trial = np.where(denom > 0, (r_vals - l_vals) / denom, 0.0)
            mean_ai = ai_vals_trial.mean()

            try:
                w_stat, p_val = stats.wilcoxon(diffs, alternative="two-sided")
            except ValueError:
                w_stat, p_val = 0.0, 1.0

            subj_results.append((subj, mean_ai, w_stat, p_val, None))

        # FDR correction across this pair's own family of subjects (insufficient-trial
        # subjects excluded from the family, same convention as elsewhere in this file).
        valid_idx = [i for i, r in enumerate(subj_results) if r[3] is not None]
        if valid_idx:
            p_values_pair = [subj_results[i][3] for i in valid_idx]
            _, p_fdr_pair, _, _ = multipletests(p_values_pair, method="fdr_bh")
            for idx, p_fdr in zip(valid_idx, p_fdr_pair):
                subj, mean_ai, w_stat, p_val, _ = subj_results[idx]
                subj_results[idx] = (subj, mean_ai, w_stat, p_val, p_fdr)

        for subj, mean_ai, w_stat, p_val, p_fdr in subj_results:
            if p_val is None:
                print(f"    {str(subj):<12} {'—':>7}  insufficient trials")
                continue
            direction = "R" if mean_ai > 0 else "L"
            sig_str = _star(p_fdr)
            color = Fore.GREEN if p_fdr < 0.05 else Fore.RED
            print(f"    {str(subj):<12} {mean_ai:>+7.4f} {w_stat:>7.1f} {p_val:>8.4f} {p_fdr:>8.4f}  "
                  f"{color}{direction} {sig_str}{Style.RESET_ALL}")

    # ── Print table ───────────────────────────────────────────────────────────
    print("\nAsymmetry Index per bilateral pair  (AI = (R−L)/(R+L))")
    print("Positive = right-dominant  |  Negative = left-dominant  |  0 = symmetric")
    header = f"  {'Pair':<18} {'Mean AI':>8} {'SD AI':>7} {'r_rb':>6} {'p_raw':>8} {'p_adj':>8}  sig"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for _, row in df_res.iterrows():
        if row["sig_fdr"]:
            sig_str = Fore.GREEN + f"✓ FDR ({_star(row['p_raw'])})" + Style.RESET_ALL
        elif row["sig_uncorr"]:
            sig_str = Fore.YELLOW + f"✓ unc ({_star(row['p_raw'])})" + Style.RESET_ALL
        else:
            sig_str = Fore.RED + "✗" + Style.RESET_ALL

        print(f"  {row['pair']:<18} {row['mean_ai']:>8.4f} {row['sd_ai']:>7.4f} "
              f"{row['r_rb']:>6.3f} {row['p_raw']:>8.4f} {row['p_adj']:>8.4f}  {sig_str}")

    dominant = df_res.loc[df_res["mean_ai"].abs().idxmax()]
    direction = "right" if dominant["mean_ai"] > 0 else "left"
    print(f"\n  Strongest lateral bias: {dominant['pair']}  "
          f"({direction}-dominant, mean AI = {dominant['mean_ai']:.4f}, "
          f"r_rb = {dominant['r_rb']:.3f})")
    print(f"\n  ⚠ N={int(df_res['n_subjects'].max())}: minimum achievable two-tailed p ≈ 0.031. "
          f"FDR correction reported but likely underpowered. "
          f"Treat AI and r_rb as primary interpretive quantities.")

    if generate_figure:
        pair_order = [
            f"{r.replace('Right', '')} R/L"
            for r, _ in BILATERAL_PAIRS
            if f"{r.replace('Right', '')} R/L" in df_res["pair"].values
        ]
        df_plot = df_res.set_index("pair").reindex(pair_order).reset_index()

        fig, ax = plt.subplots(figsize=(7, max(4, len(df_plot) * 0.55 + 1.2)))
        y_pos = np.arange(len(df_plot))

        for i, row in df_plot.iterrows():
            ai = row["mean_ai"]
            sig = row["sig_uncorr"]
            color = "#d6604d" if ai >= 0 else "#2166ac"
            alpha = 1.0 if sig else 0.40

            # Negate x so right-dominant (AI > 0) plots leftward, matching the
            # silhouette convention where the body's right side is on the left.
            ai_plot = ai

            # Error bar (±1 SD across subjects)
            ax.plot([ai_plot - row["sd_ai"], ai_plot + row["sd_ai"]], [i, i],
                    color=color, alpha=alpha * 0.75, linewidth=3.5,
                    solid_capstyle="round", zorder=2)
            # Stem
            ax.plot([0, ai_plot], [i, i], color=color, alpha=alpha,
                    linewidth=2.0, solid_capstyle="round", zorder=3)
            # Head
            ax.scatter(ai_plot, i, color=color, alpha=alpha, s=65, zorder=4)
            # Significance star (placed outside the SD bar)
            if sig:
                offset = -0.025 if ai >= 0 else 0.025  # flipped with axis
                ha = "right" if ai >= 0 else "left"
                sd_nudge = -row["sd_ai"] if ai >= 0 else row["sd_ai"]
                ax.text(ai_plot + offset + sd_nudge, i, _star(row["p_raw"]),
                        ha=ha, va="center", fontsize=9,
                        color=color, fontweight="bold")

        ax.axvline(0, color="black", linewidth=0.9, zorder=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot["pair"], fontsize=10)
        ax.invert_yaxis()

        x_bound = max(df_plot["mean_ai"].abs().max() + df_plot["sd_ai"].max() + 0.05, 0.25)
        ax.set_xlim(x_bound + 0.05, -x_bound - 0.05)  # inverted: right→left on figure
        ax.set_xlabel("← Right-dominant                    Left-dominant →",
                      fontsize=10)
        ax.set_title(
            "Bilateral symmetry — lateral dominance per joint pair\n"
            "Marker = mean AI  |  Bar = ±1 SD across subjects  "
            "|  Opacity = significance (p < .05 uncorrected)",
            fontsize=10
        )
        ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d6604d", label="Right-dominant  (AI > 0)"),
            Patch(facecolor="#2166ac", label="Left-dominant   (AI < 0)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.85)

        fig.tight_layout()
        save_figure(
            fig, PATH_FIGURES,
            f"fig{str(n).zfill(2)}_bilateral_symmetry.png",
            **SAVE_KW
        )
        plt.close(fig)

def bilateral_symmetry_by_condition(n, df, generate_figure=True):
    print_section_title(f"{n}. BILATERAL SYMMETRY — CONDITION EFFECT ON ASYMMETRY")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    c_masked, c_unmasked = "Masked", "Unmasked"

    def _star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    # ── Per-subject × condition mean speed per joint ──────────────────────────
    subj_cond_joint = (
        df_test
        .groupby(["subject", "condition", "label"])["value"]
        .mean()
        .unstack("label")
    )

    # ── ΔAI per subject for each bilateral pair ───────────────────────────────
    results  = []

    for right_j, left_j in BILATERAL_PAIRS:
        missing = [j for j in [right_j, left_j]
                   if j not in subj_cond_joint.columns]
        if missing:
            continue

        ai_by_subj_cond = {}
        for cond in [c_masked, c_unmasked]:
            if cond not in subj_cond_joint.index.get_level_values("condition"):
                continue
            cond_df = subj_cond_joint.xs(cond, level="condition")[[right_j, left_j]].dropna()
            denom   = cond_df[right_j] + cond_df[left_j]
            with np.errstate(invalid="ignore"):
                ai = np.where(denom > 0,
                              (cond_df[right_j] - cond_df[left_j]) / denom,
                              0.0)
            ai_by_subj_cond[cond] = pd.Series(ai, index=cond_df.index)

        if c_masked not in ai_by_subj_cond or c_unmasked not in ai_by_subj_cond:
            continue

        # Align subjects present in both conditions
        ai_df   = pd.DataFrame(ai_by_subj_cond).dropna()
        if len(ai_df) < 3:
            continue

        delta_ai = (ai_df[c_masked] - ai_df[c_unmasked]).values
        n_pairs  = len(delta_ai)
        mean_dai = delta_ai.mean()
        pair_label = f"{right_j.replace('Right', '')} R/L"

        try:
            w_stat, p_raw = stats.wilcoxon(delta_ai, alternative="two-sided")
        except ValueError:
            w_stat, p_raw = 0.0, 1.0

        total_ranks = n_pairs * (n_pairs + 1) / 2
        r_abs = 1.0 - (2.0 * w_stat) / total_ranks
        r_rb  = r_abs * np.sign(mean_dai) if mean_dai != 0 else 0.0

        results.append({
            "pair":             pair_label,
            "right_j":          right_j,
            "left_j":           left_j,
            "mean_dai":         mean_dai,
            "sd_dai":           delta_ai.std(),
            "mean_ai_masked":   ai_df[c_masked].mean(),
            "mean_ai_unmasked": ai_df[c_unmasked].mean(),
            "W":                w_stat,
            "p_raw":            p_raw,
            "r_rb":             r_rb,
            "n_subjects":       n_pairs,
        })

    df_res = pd.DataFrame(results)

    # ── FDR correction ────────────────────────────────────────────────────────
    reject_fdr, p_adj, _, _ = multipletests(df_res["p_raw"], method="fdr_bh")
    df_res["p_adj"]      = p_adj
    df_res["sig_fdr"]    = reject_fdr
    df_res["sig_uncorr"] = df_res["p_raw"] < 0.05

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\nΔAI = AI_masked − AI_unmasked per bilateral pair")
    print(f"Positive ΔAI = more right-dominant when masked")
    print(f"Negative ΔAI = more left-dominant when masked")
    header = (f"  {'Pair':<18} {'AI_mask':>8} {'AI_unmask':>10} "
              f"{'ΔAI':>8} {'SD':>7} {'r_rb':>6} {'p_raw':>8} {'p_adj':>8}  sig")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for _, row in df_res.iterrows():
        if row["sig_fdr"]:
            sig_str = Fore.GREEN + f"✓ FDR ({_star(row['p_raw'])})" + Style.RESET_ALL
        elif row["sig_uncorr"]:
            sig_str = Fore.YELLOW + f"✓ unc ({_star(row['p_raw'])})" + Style.RESET_ALL
        else:
            sig_str = Fore.RED + "✗" + Style.RESET_ALL

        print(f"  {row['pair']:<18} {row['mean_ai_masked']:>8.4f} "
              f"{row['mean_ai_unmasked']:>10.4f} {row['mean_dai']:>8.4f} "
              f"{row['sd_dai']:>7.4f} {row['r_rb']:>6.3f} "
              f"{row['p_raw']:>8.4f} {row['p_adj']:>8.4f}  {sig_str}")

    largest = df_res.loc[df_res["mean_dai"].abs().idxmax()]
    direction = "more right-dominant" if largest["mean_dai"] > 0 else "more left-dominant"
    print(f"\n  Largest condition shift: {largest['pair']}  "
          f"({direction} when masked, ΔAI = {largest['mean_dai']:.4f}, "
          f"r_rb = {largest['r_rb']:.3f})")
    print(f"\n  ⚠ N={int(df_res['n_subjects'].max())}: minimum achievable two-tailed "
          f"p ≈ 0.031. FDR correction reported but likely underpowered. "
          f"Treat ΔAI and r_rb as primary interpretive quantities.")

    if generate_figure:
        pair_order = [
            f"{r.replace('Right', '')} R/L"
            for r, _ in BILATERAL_PAIRS
            if f"{r.replace('Right', '')} R/L" in df_res["pair"].values
        ]
        df_plot = df_res.set_index("pair").reindex(pair_order).reset_index()

        fig, ax = plt.subplots(figsize=(7, max(4, len(df_plot) * 0.55 + 1.4)))
        y_pos = np.arange(len(df_plot))

        for i, row in df_plot.iterrows():
            dai      = row["mean_dai"]
            dai_plot = dai   # negate so right-dominant plots leftward
            sig      = row["sig_uncorr"]
            color    = "#d6604d" if dai >= 0 else "#2166ac"
            alpha    = 1.0 if sig else 0.40

            # SD bar with end ticks
            tick = 0.008
            ax.plot([dai_plot - row["sd_dai"], dai_plot + row["sd_dai"]], [i, i],
                    color=color, alpha=alpha * 0.75, linewidth=3.5,
                    solid_capstyle="round", zorder=2)
            for x_end in [dai_plot - row["sd_dai"], dai_plot + row["sd_dai"]]:
                ax.plot([x_end, x_end], [i - tick, i + tick],
                        color=color, alpha=alpha * 0.75, linewidth=1.8, zorder=2)

            # Stem + head
            ax.plot([0, dai_plot], [i, i], color=color, alpha=alpha,
                    linewidth=2.0, solid_capstyle="round", zorder=3)
            ax.scatter(dai_plot, i, color=color, alpha=alpha, s=65, zorder=4)

            # Significance star (outside SD bar, offsets flipped with axis)
            if sig:
                offset   = -0.008 if dai >= 0 else 0.008
                ha       = "right" if dai >= 0 else "left"
                sd_nudge = -row["sd_dai"] if dai >= 0 else row["sd_dai"]
                ax.text(dai_plot + sd_nudge + offset, i, _star(row["p_raw"]),
                        ha=ha, va="center", fontsize=9,
                        color=color, fontweight="bold")

        ax.axvline(0, color="black", linewidth=0.9, zorder=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot["pair"], fontsize=10)
        ax.invert_yaxis()

        x_bound = max(df_plot["mean_dai"].abs().max() + df_plot["sd_dai"].max() + 0.02, 0.15)
        ax.set_xlim(x_bound + 0.02, -x_bound - 0.02)   # inverted: right→left on figure
        ax.set_xlabel(
            "← More right-dominant when masked          "
            "More left-dominant when masked →",
            fontsize=10
        )
        ax.set_title(
            f"Condition effect on bilateral asymmetry  (Masked vs Unmasked)\n"
            f"Marker = mean ΔAI  |  Bar = ±1 SD  "
            f"|  Opacity = significance (p < .05 uncorrected)",
            fontsize=10
        )
        ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d6604d", label="More right-dominant when masked  (ΔAI > 0)"),
            Patch(facecolor="#2166ac", label="More left-dominant when masked   (ΔAI < 0)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.85)

        fig.tight_layout()
        save_figure(
            fig, PATH_FIGURES,
            f"fig{str(n).zfill(2)}_bilateral_symmetry_by_condition.png",
            **SAVE_KW
        )
        plt.close(fig)


def joint_correlation_matrix(n, df, generate_figure=True):
    print_section_title(f"{n}. JOINT-TO-JOINT CORRELATION MATRIX")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    joints_test = [j for j in joints if j != "SpineBase"] if USE_REREF else joints.copy()
    n_joints    = len(joints_test)

    trial_speed = (df_test[df_test["label"].isin(joints_test)].groupby(["subject", "trial", "label"])["value"].mean()
                   .unstack("label").reindex(columns=joints_test))

    subj_z_matrices = []

    print("\nPer-subject Spearman correlation matrices (Fisher z-averaged):")
    for subj in subjects:
        if subj not in trial_speed.index.get_level_values("subject"):
            continue
        subj_data = trial_speed.xs(subj, level="subject").dropna(how="all")

        # Spearman correlation matrix (joints × joints)
        r_mat, _ = spearmanr(subj_data.values)
        if r_mat.ndim == 0:
            continue
        r_mat = np.array(r_mat)

        # Clip to avoid Fisher z ±inf at exact ±1
        r_mat_clipped = np.clip(r_mat, -0.9999, 0.9999)
        z_mat = np.arctanh(r_mat_clipped)
        subj_z_matrices.append(z_mat)
        print(f"  {subj}: {subj_data.shape[0]} trials used")

    z_mean  = np.mean(subj_z_matrices, axis=0)
    r_group = np.tanh(z_mean)
    np.fill_diagonal(r_group, 1.0)

    df_r = pd.DataFrame(r_group, index=joints_test, columns=joints_test)

    n_permutations = 1000
    rng = np.random.default_rng(42)

    print(f"\nRunning permutation test ({n_permutations} permutations)...")

    # Observed upper-triangle indices
    triu_idx = np.array([(i, j)
                         for i in range(n_joints)
                         for j in range(i + 1, n_joints)])

    null_dist = np.zeros((n_permutations, len(triu_idx)))

    for perm in range(n_permutations):
        perm_z_matrices = []
        for subj in subjects:
            if subj not in trial_speed.index.get_level_values("subject"):
                continue
            subj_data = trial_speed.xs(subj, level="subject").dropna(how="all").values
            # Shuffle trial order independently per subject
            shuffled = np.apply_along_axis(rng.permutation, 0, subj_data)
            r_perm, _ = spearmanr(shuffled)
            r_perm = np.clip(np.array(r_perm), -0.9999, 0.9999)
            perm_z_matrices.append(np.arctanh(r_perm))

        z_perm  = np.mean(perm_z_matrices, axis=0)
        r_perm  = np.tanh(z_perm)
        null_dist[perm] = [r_perm[i, j] for i, j in triu_idx]

    # Two-tailed p: proportion of null |r| >= observed |r|
    obs_r   = np.array([r_group[i, j] for i, j in triu_idx])
    p_values = np.mean(np.abs(null_dist) >= np.abs(obs_r), axis=0)

    # FDR correction
    reject_fdr, p_adj, _, _ = multipletests(p_values, method="fdr_bh")

    # Build p-value and significance matrices
    p_mat   = pd.DataFrame(np.nan, index=joints_test, columns=joints_test)
    sig_mat = pd.DataFrame(False,  index=joints_test, columns=joints_test)
    for (i, j), p_raw, p_a, sig in zip(triu_idx, p_values, p_adj, reject_fdr):
        ji, jj = joints_test[i], joints_test[j]
        p_mat.loc[ji, jj] = p_mat.loc[jj, ji] = p_raw
        sig_mat.loc[ji, jj] = sig_mat.loc[jj, ji] = sig

    # ── Print summary ─────────────────────────────────────────────────────────
    n_sig = reject_fdr.sum()
    print(f"\nSignificant pairs (FDR-BH, permutation): {n_sig}/{len(triu_idx)}")

    print("\nStrongest positive correlations (FDR-corrected):")
    def _sort_key(x):
        return -x[2] if np.isfinite(x[2]) else float('inf')

    pair_summary = sorted(
        [(joints_test[i], joints_test[j], obs_r[k], p_values[k], p_adj[k], reject_fdr[k])
         for k, (i, j) in enumerate(triu_idx)],
        key=_sort_key
    )

    print("\nTop 50 strongest negative correlations (FDR-corrected):")
    for j1, j2, r_val, p_raw, p_a, sig in pair_summary[:50]:
        sym = Fore.GREEN + "✓" + Style.RESET_ALL if sig else Fore.RED + "✗" + Style.RESET_ALL
        print(f"{(j1 + " × " + j2):<40}  r = {r_val:+.3f} p_raw = {p_raw:.4f} p_adj = {p_a:.4f} {sym}")

    print("\nTop 10 strongest negative correlations (FDR-corrected):")
    for j1, j2, r_val, p_raw, p_a, sig in sorted(pair_summary, key=lambda x: x[2])[:10]:
        sym = Fore.GREEN + "✓" + Style.RESET_ALL if sig else Fore.RED + "✗" + Style.RESET_ALL
        print(f"  {j1:<22} × {j2:<22}  r = {r_val:+.3f}  p_raw = {p_raw:.4f}  "
              f"p_adj = {p_a:.4f}  {sym}")

    if generate_figure:
        # Build anatomical ordering from JOINT_GROUPS
        ordered = []
        for group_joints in JOINT_GROUPS.values():
            ordered.extend([j for j in group_joints if j in joints_test])
        ordered.extend([j for j in joints_test if j not in ordered])

        df_r_ord = df_r.reindex(index=ordered, columns=ordered)
        sig_mat_ord = sig_mat.reindex(index=ordered, columns=ordered)
        n_ord = len(ordered)

        # Group boundary positions
        group_boundaries = []
        pos = 0
        group_label_positions = {}
        for group_name, group_joints in JOINT_GROUPS.items():
            valid = [j for j in group_joints if j in joints_test]
            if not valid:
                continue
            group_label_positions[group_name] = pos + len(valid) / 2 - 0.5
            pos += len(valid)
            group_boundaries.append(pos)
        group_boundaries = group_boundaries[:-1]  # drop last (no line needed after final group)

        cell_size = 0.55  # inches per cell
        fig_size = n_ord * cell_size + 3.5
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Diverging colormap: blue=negative, white=0, red=positive
        cmap = LinearSegmentedColormap.from_list(
            "rmat", ["#2166ac", "#f7f7f7", "#d6604d"], N=256)

        sns.heatmap(
            df_r_ord,
            ax=ax,
            cmap=cmap,
            vmin=-1, vmax=1,
            center=0,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Spearman r (Fisher-averaged)", "shrink": 0.6},
            xticklabels=True,
            yticklabels=True,
            annot=False,  # custom annotation below
        )

        # Annotate each cell with r value; hatch non-significant off-diagonal cells
        for row_i, ji in enumerate(ordered):
            for col_i, jj in enumerate(ordered):
                r_val = df_r_ord.loc[ji, jj]
                is_sig = sig_mat_ord.loc[ji, jj]
                is_diag = (ji == jj)

                label = "1.00" if is_diag else f"{r_val:+.2f}"
                text_color = "white" if abs(r_val) > 0.55 else "black"
                ax.text(col_i + 0.5, row_i + 0.5, label,
                        ha="center", va="center",
                        fontsize=6.5, color=text_color)

                if not is_diag and not is_sig:
                    ax.add_patch(plt.Rectangle(
                        (col_i, row_i), 1, 1,
                        fill=False, hatch="////",
                        edgecolor="black", linewidth=0,
                        alpha=0.25, zorder=3
                    ))

        # Group boundary lines
        for boundary in group_boundaries:
            ax.axhline(boundary, color="black", linewidth=1.8, zorder=4)
            ax.axvline(boundary, color="black", linewidth=1.8, zorder=4)

        # Group labels: above x-axis ticks and to the left of y-axis ticks
        # Using axis-fraction coordinates so no twin axes are needed
        for group_name, mid in group_label_positions.items():
            # Top: transform x from data coords, y just above axes
            ax.text((mid + 0.5) / n_ord, 1.012, group_name,
                    ha="center", va="bottom",
                    fontsize=8, fontstyle="italic", color="#333",
                    transform=ax.transAxes)
            # Left: x just left of axes, y from data coords (inverted)
            ax.text(-0.18, 1.0 - (mid + 0.5) / n_ord, group_name,
                    ha="center", va="center",
                    fontsize=8, fontstyle="italic", color="#333",
                    rotation=90,
                    transform=ax.transAxes)

        ax.set_title(
            "Joint-to-joint speed correlation matrix\n"
            "Fisher-averaged Spearman r across subjects  |  "
            "Hatched = not significant (permutation test, FDR-BH)",
            fontsize=11, pad=28
        )
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

        fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)
        save_figure(fig, PATH_FIGURES,
                    f"fig{str(n).zfill(2)}_joint_correlation_matrix.png",
                    **SAVE_KW)
        plt.close(fig)

    return df_r


def joint_correlation(n, df, target_joint, generate_figure=True):
    print_section_title(f"{n}. JOINT CORRELATION — {target_joint.upper()}")

    if USE_REREF:
        df_test = df[df["label"] != "SpineBase"].copy()
    else:
        df_test = df.copy()

    joints_test = [j for j in joints if j != "SpineBase"] if USE_REREF else joints.copy()

    if target_joint not in joints_test:
        raise ValueError(f"'{target_joint}' not found in joints. "
                         f"Available: {joints_test}")

    target_idx = joints_test.index(target_joint)
    other_idx = [i for i in range(len(joints_test)) if i != target_idx]
    other_joints = [joints_test[i] for i in other_idx]

    # ── Trial-level mean speed: subject × trial × joint ──────────────────────
    trial_speed = (
        df_test[df_test["label"].isin(joints_test)]
        .groupby(["subject", "trial", "label"])["value"]
        .mean()
        .unstack("label")
        .reindex(columns=joints_test)
    )

    # Cache per-subject data arrays for permutation reuse
    subj_arrays = []
    for subj in subjects:
        if subj not in trial_speed.index.get_level_values("subject"):
            continue
        subj_data = trial_speed.xs(subj, level="subject").dropna(how="all").values
        subj_arrays.append(subj_data)

    def _fisher_avg_row(arrays):
        """Fisher-average the target joint's correlation row across subjects."""
        z_vecs = []
        for data in arrays:
            r_mat, _ = spearmanr(data)
            r_row = np.clip(np.array(r_mat)[target_idx], -0.9999, 0.9999)
            z_vecs.append(np.arctanh(r_row))
        return np.tanh(np.mean(z_vecs, axis=0))

    # ── Observed correlations ─────────────────────────────────────────────────
    r_group = _fisher_avg_row(subj_arrays)
    corr = pd.Series(r_group, index=joints_test)
    corr[target_joint] = 1.0
    obs_r = np.array([r_group[i] for i in other_idx])

    # ── Permutation test (column-wise shuffle, target joint's row only) ───────
    n_permutations = 1000
    print(f"Running permutation test ({n_permutations} permutations)...")
    rng = np.random.default_rng(42)
    null_dist = np.zeros((n_permutations, len(other_idx)))

    for perm in range(n_permutations):
        perm_arrays = [
            np.apply_along_axis(rng.permutation, 0, data)
            for data in subj_arrays
        ]
        r_perm = _fisher_avg_row(perm_arrays)
        null_dist[perm] = [r_perm[i] for i in other_idx]

    # Two-tailed p: proportion of null |r| >= observed |r|
    p_values = np.mean(np.abs(null_dist) >= np.abs(obs_r), axis=0)

    # FDR correction
    reject_fdr, p_adj, _, _ = multipletests(p_values, method="fdr_bh")

    # Build result series aligned to other_joints
    p_raw_s = pd.Series(p_values, index=other_joints)
    p_adj_s = pd.Series(p_adj, index=other_joints)
    sig_s = pd.Series(reject_fdr, index=other_joints)

    # ── Print ─────────────────────────────────────────────────────────────────
    def _star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    print(f"\nSpearman r (Fisher-averaged) with {target_joint}:")
    header = f"  {'Joint':<22} {'r':>6} {'p_raw':>8} {'p_adj':>8}  sig"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for joint, r_val in corr.drop(index=target_joint).sort_values(ascending=False).items():
        p_r = p_raw_s[joint]
        p_a = p_adj_s[joint]
        sig = sig_s[joint]
        if sig:
            sig_str = Fore.GREEN + f"✓ FDR ({_star(p_r)})" + Style.RESET_ALL
        elif p_r < 0.05:
            sig_str = Fore.YELLOW + f"✓ unc ({_star(p_r)})" + Style.RESET_ALL
        else:
            sig_str = Fore.RED + "✗" + Style.RESET_ALL
        print(f"  {joint:<22} {r_val:>+6.3f} {p_r:>8.4f} {p_a:>8.4f}  {sig_str}")

    if generate_figure:
        # Build dict for all joints; target joint gets 1.0 (diagonal)
        r_dict = corr.to_dict()

        # SpineBase excluded from test — set to 0 if present
        if USE_REREF and "SpineBase" in joints:
            r_dict.setdefault("SpineBase", 0.0)

        # Ensure all joints in the global list have a value
        for j in joints:
            r_dict.setdefault(j, 0.0)

        r_max = max(abs(v) for k, v in r_dict.items() if k != target_joint)
        r_max = max(r_max, 0.1)

        plot_silhouette(
            r_dict,
            title=f"Correlation with {target_joint}\n(Fisher-averaged Spearman r)",
            title_scale=f"Spearman r  (ref: {target_joint})",
            min_scale=0,
            max_scale=r_max,
            color_scheme="celsius",
            show_values=True,
            value_format=".2f",
            strip_leading_zero=True,
            show=False,
            path_save=op.join(PATH_FIGURES,
                              f"fig{str(n).zfill(2)}_joint_correlation_{target_joint}.png"),
            resolution=(800, 1080),
            verbosity=0,
        )


def _annotate_reref_axis(fig, joint_name="SpineBase", hatch="////", text="Re-referencing joint"):
    import matplotlib.patches as mpatches

    for ax in fig.axes:
        if ax.get_title() == joint_name:
            rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, hatch=hatch, facecolor="white",
                                      edgecolor="lightgray", linewidth=0.5, zorder=0)
            ax.add_patch(rect)
            ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontstyle="italic",
                    fontsize=7, color="gray", zorder=2)
            break


def joint_coherence_body_graph(n, target_joint, sequence_measure, value_type="z-score", level="subject",
                               gesture_band=None, generate_figure=True):
    from krajjat.classes.graph_element import Graph
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    trailing_file_name = REREF_SUFFIX if USE_REREF else ""

    level_label = "PER SUBJECT" if level == "subject" else "GROUP"
    print_section_title(f"{n}. JOINT COHERENCE BODY GRAPH ({level_label}) — {target_joint.upper()}")

    # Always load per-subject results (the _avgtrial files) -- "group" is computed by
    # aggregating these below, rather than loading the separate _concat file, which
    # turned out not to reflect an actual multi-subject aggregate.
    results_by_subject = _load_coherence_results(sequence_measure, target_joint, level="subject")
    if not results_by_subject:
        return

    # Extract each subject's data first, regardless of level.
    per_subject_data = {}  # subject -> data_by_joint
    per_subject_pvals = {}  # subject -> p_by_joint (z-score mode only)
    freqs = None
    lag = None
    out_ref = None
    for subject, out in results_by_subject.items():
        if out_ref is None:
            out_ref = out
        data_by_joint, p_by_joint, lag, freqs = _extract_by_joint(out, target_joint, sequence_measure, value_type)
        per_subject_data[subject] = data_by_joint
        if value_type == "z-score":
            per_subject_pvals[subject] = p_by_joint

    subject_names_all = list(per_subject_data.keys())
    all_joints = list(next(iter(per_subject_data.values())).keys())

    if level == "subject":
        all_data = per_subject_data
        all_pvals = per_subject_pvals
        subject_names = subject_names_all
    else:
        # level == "group": aggregate each joint's array across subjects.
        n_subjects = len(subject_names_all)
        all_data = {"Group": {}}
        all_pvals = {"Group": {}} if value_type == "z-score" else {}
        for joint in all_joints:
            all_data["Group"][joint] = {}
            if value_type == "z-score":
                all_pvals["Group"][joint] = {}
            for lag_key in per_subject_data[subject_names_all[0]][joint]:
                stacked = np.stack([per_subject_data[s][joint][lag_key] for s in subject_names_all], axis=0)
                if value_type == "raw":
                    # Raw coherence: a plain mean across subjects is the correct aggregate
                    # (matches joint_coherence_silhouette's level="subject" averaging).
                    all_data["Group"][joint][lag_key] = np.nanmean(stacked, axis=0)
                else:
                    # z-score: Stouffer's method (sum of per-subject z / sqrt(N)), NOT a
                    # plain mean -- consistent with the sqrt(N) z-score inflation fix already
                    # applied elsewhere in this pipeline (averaging z-scores directly
                    # under- or over-states combined significance).
                    combined_z = np.nansum(stacked, axis=0) / np.sqrt(n_subjects)
                    all_data["Group"][joint][lag_key] = combined_z
                    # One-tailed p-value from the combined z (assumes positive z = more
                    # coherent than chance, matching the existing per-subject convention --
                    # adjust to two-tailed here if that assumption doesn't hold).
                    all_pvals["Group"][joint][lag_key] = norm.sf(combined_z)
        subject_names = ["Group"]

    print(f"\n  Found {len(subject_names_all)} entry/entries: {subject_names_all}")

    # ── boundary z's for the significance threshold (z-score mode only) ────────
    boundary_zs = []
    if value_type == "z-score":
        for name in all_data:
            for joint in all_data[name]:
                z_arr = all_data[name][joint][lag]
                p_arr = all_pvals[name][joint][lag]
                if np.all(np.isnan(z_arr)):
                    continue
                nonsig_mask = p_arr >= out_ref.signif_alpha
                if nonsig_mask.any():
                    boundary_zs.append(float(z_arr[nonsig_mask].max()))

    value_label = "z-scores" if value_type == "z-score" else "coherence"
    z_threshold = float(np.median(boundary_zs)) if boundary_zs else 1.96

    if value_type == "z-score":
        print(f"\n  Significance threshold z = {z_threshold:.3f}  "
              f"(α = {out_ref.signif_alpha}, n_perms = {out_ref.number_of_randperms})")

    # ═══════════════════════════════════════════════════════════════════════════
    # EVERYTHING BELOW THIS LINE IS UNCHANGED FROM YOUR ORIGINAL FUNCTION.
    # It already treats "Group" as just another name in all_data/subject_names
    # (that's how the old, buggy single-file group case worked too), so no edits
    # were needed here -- paste your original console-summary/plotting code below
    # this point, verbatim.
    # ═══════════════════════════════════════════════════════════════════════════

    # ── console summary ────────────────────────────────────────────────────────
    print(f"\n  Coherence with {target_joint}  |  freq resolution: {out_ref.freq_resolution_hz} Hz")
    nonzero_mask = (freqs > 0) & (freqs <= 4)  # excludes the 0 Hz (DC) bin from peak detection

    for joint in all_joints:
        print(f"\n  {joint}")
        if value_type == "z-score":
            print(f"    {'Subject':<20} {'Peak z':>8}  {'@ Hz':>7}  {'Sig bins':>9}")
            print("    " + "─" * 50)
            for name in subject_names:
                z_arr = all_data[name][joint][lag]
                p_arr = all_pvals[name][joint][lag]
                if np.all(np.isnan(z_arr)):
                    continue
                z_search = np.where(nonzero_mask, z_arr, np.nan)
                if np.all(np.isnan(z_search)):
                    continue
                peak_idx = int(np.nanargmax(z_search))
                n_sig = int(np.sum(p_arr < out_ref.signif_alpha))
                print(f"    {name:<20} {z_arr[peak_idx]:>8.2f}  {freqs[peak_idx]:>7.3f}  {n_sig:>9}")
        else:
            print(f"    {'Subject':<20} {'Peak coh':>10}  {'@ Hz':>7}")
            print("    " + "─" * 44)
            for name in subject_names:
                arr = all_data[name][joint][lag]
                if np.all(np.isnan(arr)):
                    continue
                arr_search = np.where(nonzero_mask, arr, np.nan)
                if np.all(np.isnan(arr_search)):
                    continue
                peak_idx = int(np.nanargmax(arr_search))
                print(f"    {name:<20} {arr[peak_idx]:>10.4f}  {freqs[peak_idx]:>7.3f}")

    if generate_figure:
        # ── build plot_dict: one Graph per joint, one line per subject (or one line for the group) ──
        plot_dict = {}
        for joint in all_joints:
            g = Graph()
            if level == "subject":
                for subject in ALL_SUBJECTS:
                    if subject not in all_data:
                        continue
                    arr = all_data[subject][joint][lag]
                    if np.all(np.isnan(arr)):
                        continue
                    g.add_plot(freqs, arr, color=SUBJECT_COLORS[subject], line_width=1, label=subject)
            else:
                arr = all_data["Group"][joint][lag]
                if not np.all(np.isnan(arr)):
                    g.add_plot(freqs, arr, color="black", line_width=1.5, label="Group")
            plot_dict[joint] = g

        if value_type == "z-score":
            extra_kwargs = dict(overlay_lines=z_threshold, overlay_lines_color="red", overlay_lines_style="--",
                overlay_lines_width=1.5, )
        else:
            extra_kwargs = dict(overlay_lines=None, ylim=[-0.01, 1.01], )

        path_png = op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_joint_coherence_body_graph_"
                                         f"{target_joint}{trailing_file_name}_{value_type}_{level}.png")

        mpl_kwargs = {"hspace": 0.3, "tick_pad": -2, "xlabel_pad": -1, "ylabel_pad": 0, "title_pad": 3,
                      "title_weight": "bold", "xtick_step": 1, "tight_layout": False}

        joint_layout = [[None, None, "Head", None, None], [None, None, "Neck", None, None],
                        ["ElbowRight", "ShoulderRight", "SpineShoulder", "ShoulderLeft", "ElbowLeft"],
                        ["WristRight", None, "SpineMid", None, "WristLeft"],
                        ["HandRight", "HipRight", "SpineBase", "HipLeft", "HandLeft"]]

        fig = plot_body_graphs(plot_dict, title=None, sns_kwargs={"style": "whitegrid",
                                                                  "rc": {"font.family": "Metropolis",
                                                                         "axes.spines.right": False,
                                                                         "axes.spines.top": False}},
            mpl_kwargs=mpl_kwargs, xlim=[0, 4], joint_layout=joint_layout, legend=False, show=False, path_save=None,
            **extra_kwargs, )
        _annotate_reref_axis(fig)

        fig.savefig(path_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved body graph: {path_png}")


def _aggregate_subject_coherence(target_joint, sequence_measure, value_type):
    """Load each subject's coherence result and aggregate across subjects.

    value_type="raw"     -> plain mean across subjects.
    value_type="z-score" -> Stouffer's method (sum(z) / sqrt(N)), NOT a plain mean --
        consistent with the sqrt(N) z-score inflation fix already applied elsewhere
        in this pipeline (averaging z-scores directly under/overstates combined
        significance).

    Returns (data_by_joint, lag, freqs, subject_names), or (None, None, None, [])
    if no subject files were found. data_by_joint[joint][lag] is the aggregated
    array across all subjects found on disk for this joint.
    """
    results_by_subject = _load_coherence_results(sequence_measure, target_joint, level="subject")
    if not results_by_subject:
        return None, None, None, []

    stacked = {}
    lag = None
    freqs = None
    for subject, out in results_by_subject.items():
        subj_data_by_joint, _, lag, freqs = _extract_by_joint(out, target_joint, sequence_measure, value_type)
        for joint, arr_by_lag in subj_data_by_joint.items():
            stacked.setdefault(joint, []).append(arr_by_lag[lag])

    n_subjects = len(results_by_subject)
    if value_type == "z-score":
        data_by_joint = {joint: {lag: np.nansum(arrays, axis=0) / np.sqrt(n_subjects)} for joint, arrays in
                         stacked.items() if len(arrays) > 0}
    else:
        data_by_joint = {joint: {lag: np.nanmean(arrays, axis=0)} for joint, arrays in stacked.items() if
                         len(arrays) > 0}

    return data_by_joint, lag, freqs, list(results_by_subject.keys())


def joint_coherence_silhouette(n, target_joint, sequence_measure, frequency=None, value_type="raw", level="group",
                               generate_figure=True):
    trailing_file_name = REREF_SUFFIX if USE_REREF else ""

    level_label = "GROUP" if level == "group" else "PER SUBJECT (averaged)"
    print_section_title(f"{n}. JOINT COHERENCE — {target_joint.upper()} ({level_label})")

    # Both "group" and "subject" now go through the same aggregation -- "group" IS
    # "averaged across all subjects", computed via the (correct) per-subject _avgtrial
    # files rather than the separate _concat file.
    data_by_joint, lag, freqs, subject_names = _aggregate_subject_coherence(target_joint, sequence_measure, value_type)
    if data_by_joint is None:
        return

    print(f"\n  Found {len(subject_names)} subject(s): {subject_names}")
    agg_method = "Stouffer's method" if value_type == "z-score" else "mean"
    print(f"  Aggregated across subjects using {agg_method}")

    data_by_joint = {joint: arr_by_lag for joint, arr_by_lag in data_by_joint.items() if joint in UPPER_BODY_JOINTS_REREF}

    # ═══════════════════════════════════════════════════════════════════════════
    # EVERYTHING BELOW THIS LINE IS UNCHANGED FROM YOUR ORIGINAL FUNCTION.
    # ═══════════════════════════════════════════════════════════════════════════

    value_label = "z-score" if value_type == "z-score" else "coherence"
    scale_title = "Z-score" if value_type == "z-score" else "Coherence"
    valid_joints = [j for j in data_by_joint if not np.all(np.isnan(data_by_joint[j][lag]))]

    # ── build freq_specs: list of (display_label, indices_array | None) ────────
    # indices_array = None  → mean across ALL frequencies
    # indices_array = array → mean across those specific bins (single or range)

    def _closest_idx(f):
        return int(np.argmin(np.abs(freqs - f)))

    def _range_indices(lo, hi):
        return np.where((freqs >= lo) & (freqs <= hi))[0]

    def _range_label(indices):
        return f"{freqs[indices[0]]:.2f}–{freqs[indices[-1]]:.2f} Hz"

    def _label_to_suffix(label):
        return (label.replace(" Hz", "hz").replace("–", "_to_").replace(".", "_").replace(" ", "_"))

    def _parse_freq_entry(f):
        if isinstance(f, tuple):
            lo, hi = f
            idx = _range_indices(lo, hi)
            return _range_label(idx), idx
        else:
            idx = np.array([_closest_idx(f)])
            return f"{freqs[idx[0]]:.3f} Hz", idx

    if frequency is None:
        freq_specs = [("mean", None)]
    elif isinstance(frequency, tuple):
        freq_specs = [_parse_freq_entry(frequency)]
    elif isinstance(frequency, (int, float)):
        freq_specs = [_parse_freq_entry(frequency)]
    else:
        freq_specs = [_parse_freq_entry(f) for f in frequency]

    # ── val_dict builder ───────────────────────────────────────────────────────
    def _build_val_dict(indices):
        val_dict = {}
        for joint in data_by_joint:
            arr = data_by_joint[joint][lag]
            if np.all(np.isnan(arr)):
                val_dict[joint] = 0.0
            elif indices is None:
                val_dict[joint] = float(np.nanmean(arr))
            else:
                val_dict[joint] = float(np.nanmean(arr[indices]))
        for j in UPPER_BODY_JOINTS_REREF:
            val_dict.setdefault(j, 0.0)
        return val_dict

    # ── print summary ──────────────────────────────────────────────────────────
    print(f"\n{value_label.capitalize()} — coherence with {target_joint}:")
    for label, indices in freq_specs:
        val_dict = _build_val_dict(indices)
        print(f"\n  @ {label}:")
        print(f"  {'Joint':<22} {value_label.capitalize():>10}")
        print("  " + "─" * 34)
        for joint, val in sorted([(j, val_dict[j]) for j in valid_joints if j != target_joint], key=lambda x: -x[1]):
            bar = "█" * min(int(val / 10), 30)
            print(f"  {joint:<22} {val:>10.2f}  {bar}")
        print(f"  {target_joint:<22} {'(ref)':>10}")

    if generate_figure:
        if len(freq_specs) == 1:
            label, indices = freq_specs[0]
            val_dict = _build_val_dict(indices)

            other_vals = [v for j, v in val_dict.items() if j != target_joint]
            val_max = max((v for v in other_vals if np.isfinite(v)), default=1.0)
            val_max = max(val_max, 1.0)

            freq_suffix = "mean" if indices is None else _label_to_suffix(label)
            title_freq = (
                f"mean {value_label} (all frequencies)" if indices is None else f"mean {value_label} ({label})" if len(
                    indices) > 1 else f"{value_label} @ {label}")

            plot_silhouette(val_dict, title=None, title_scale=scale_title, min_scale=0, max_scale=val_max,
                color_scheme="celsius", show_values=True, value_format=".2f", strip_leading_zero=True, font_size=0.8,
                show=False, path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_joint_coherence_{target_joint}"
                                                            f"_{freq_suffix}{trailing_file_name}_{value_type}_{level}.png"),
                path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"), resolution=(600, 1080), verbosity=0, )
            print(f"  Saved silhouette: {freq_suffix}")

        else:
            val_grid = {j: np.array([_build_val_dict(indices)[j] for _, indices in freq_specs]) for j in
                        UPPER_BODY_JOINTS_REREF}

            all_vals = [v for j in val_grid for v in val_grid[j] if j != target_joint and np.isfinite(v)]
            val_max = max(all_vals, default=1.0)
            val_max = max(val_max, 1.0)

            title_silhouettes = [label for label, _ in freq_specs]

            first_label = freq_specs[0][0]
            last_label = freq_specs[-1][0]
            freq_suffix = ("mean" if freq_specs[0][
                                         1] is None else f"{_label_to_suffix(first_label)}_to_{_label_to_suffix(last_label)}")

            n_cols = min(len(freq_specs), 4)

            plot_silhouette(val_grid, title=f"Coherence with {target_joint} — {value_label}s by frequency",
                title_silhouette=title_silhouettes, title_scale=scale_title, min_scale=0, max_scale=val_max,
                n_cols=n_cols, color_scheme="celsius", show_values=True, value_format=".1f", font_size=0.8, show=False,
                path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_joint_coherence_{target_joint}"
                                                f"_{freq_suffix}{trailing_file_name}_{value_type}_{level}.png"),
                resolution=(800 + 200 * n_cols, 1080), verbosity=0, )
            print(f"  Saved grid silhouette: {len(freq_specs)} frequencies")


def test_coherence_at_frequency(target_joint, test_joint, sequence_measure, frequency, lag=0.0, alpha=0.05):
    from krajjat import Results
    import glob
    from scipy import stats

    if USE_REREF:
        trailing_file_name = "_rSpineBase"
    else:
        trailing_file_name = ""

    coherence_dir = op.join(PATH_RESULTS, "Coherence", "Joint")
    pattern = op.join(coherence_dir,
                      f"coherence_*_{sequence_measure}_{target_joint}{trailing_file_name}.json")
    subject_files = sorted(glob.glob(pattern))

    prefix = f"coherence_"
    suffix = f"_{sequence_measure}_{target_joint}{trailing_file_name}.json"
    subject_names = [op.basename(f)[len(prefix):-len(suffix)] for f in subject_files]

    key = (target_joint, sequence_measure)
    coh_values = []
    actual_freq = None

    for subject, path_json in zip(subject_names, subject_files):
        out = Results.load(path_json)
        sm_key = out.sequence_measures[0]
        sv     = out.series_values[0]
        freqs  = np.array(out.frequencies)

        freq_idx = int(np.argmin(np.abs(freqs - frequency)))
        actual_freq = freqs[freq_idx]

        arr = out.averages[key][sm_key][sv][test_joint][lag]
        coh_values.append(float(arr[freq_idx]))

    coh_values = np.array(coh_values)
    z_values   = np.arctanh(coh_values)

    t_stat, p_value = stats.ttest_1samp(z_values, popmean=0, alternative="greater")
    mean_coh = float(np.tanh(np.mean(z_values)))

    print(f"\nCoherence of {test_joint} with {target_joint} @ {actual_freq:.3f} Hz  (lag = {lag} s)")
    print(f"  {'Subject':<20} {'Coherence':>10}  {'Fisher z':>10}")
    print("  " + "─" * 44)
    for name, c, z in zip(subject_names, coh_values, z_values):
        print(f"  {name:<20} {c:>10.4f}  {z:>10.4f}")
    print("  " + "─" * 44)
    print(f"  {'Mean (back-transformed)':<20} {mean_coh:>10.4f}")
    print(f"\n  One-sample t-test (H₀: z = 0, one-tailed greater)")
    print(f"  t({len(subject_names) - 1}) = {t_stat:.3f},  p = {p_value:.4f}  "
          f"({'*' if p_value < alpha else 'ns'}  α = {alpha})")

    return coh_values, z_values, t_stat, p_value

def joint_coherence_ttest_body_graph(n, target_joint, sequence_measure, value_type="raw", level="subject",
                                     alpha=0.05, gesture_band=None, generate_figure=True):
    from krajjat.classes.graph_element import Graph
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    trailing_file_name = REREF_SUFFIX if USE_REREF else ""

    print_section_title(f"{n}. JOINT COHERENCE T-VALUES BODY GRAPH — {target_joint.upper()}")

    if level != "subject":
        print("\n  This analysis requires per-subject variability (one-sample t-test across subjects) "
              "and is not meaningful at the group level — pass level='subject'.")
        return

    results_by_subject = _load_coherence_results(sequence_measure, target_joint, level="subject")
    if not results_by_subject:
        return

    subject_names = list(results_by_subject.keys())
    n_subjects = len(subject_names)
    print(f"\n  Found {n_subjects} subject(s): {subject_names}")

    # ── load all subjects ──────────────────────────────────────────────────────
    stacked = {}   # joint → (n_subjects, n_freqs) array
    out_ref = None
    lag = None

    for subject, out in results_by_subject.items():
        if out_ref is None:
            out_ref = out

        data_by_joint, _, lag, freqs = _extract_by_joint(out, target_joint, sequence_measure, value_type)
        for joint in UPPER_BODY_JOINTS:
            if joint not in data_by_joint:
                continue
            arr = data_by_joint[joint][lag]
            stacked.setdefault(joint, []).append(arr)

    freqs     = np.array(out_ref.frequencies)
    n_freqs   = len(freqs)
    df        = n_subjects - 1
    t_crit    = float(stats.t.ppf(1 - alpha, df=df))

    # ── compute t-test per (joint × freq bin) ─────────────────────────────────
    # raw     → Fisher-z transform of coherence, tested against 0
    # z-score → permutation z-scores tested directly against 0 (already null-centred)
    valid_joints = [j for j in UPPER_BODY_JOINTS if j in stacked and j != target_joint
                    and len(stacked[j]) == n_subjects
                    and not np.all(np.isnan(stacked[j]))]

    t_by_joint = {}
    p_by_joint = {}

    for joint in valid_joints:
        value_matrix = np.array(stacked[joint])                  # (n_subjects, n_freqs)
        if value_type == "raw":
            value_matrix = np.clip(value_matrix, 0, 1 - 1e-8)   # avoid arctanh(1) = inf
            value_matrix = np.arctanh(value_matrix)              # Fisher z transform
        t_arr      = np.full(n_freqs, np.nan)
        p_arr      = np.full(n_freqs, np.nan)
        for fi in range(n_freqs):
            z_col = value_matrix[:, fi]
            if np.any(np.isfinite(z_col)):
                t_val, p_val = stats.ttest_1samp(z_col, popmean=0, alternative="greater")
                t_arr[fi] = t_val
                p_arr[fi] = p_val
        t_by_joint[joint] = t_arr
        p_by_joint[joint] = p_arr

    # ── FDR-BH correction across all (joint × freq bin) ───────────────────────
    joint_order  = list(t_by_joint.keys())
    all_p        = np.concatenate([p_by_joint[j] for j in joint_order])
    finite_mask  = np.isfinite(all_p)
    reject       = np.zeros(len(all_p), dtype=bool)

    if finite_mask.any():
        reject[finite_mask] = multipletests(all_p[finite_mask], alpha=alpha, method="fdr_bh")[0]

    reject_by_joint = {}
    idx = 0
    for joint in joint_order:
        n = len(p_by_joint[joint])
        reject_by_joint[joint] = reject[idx:idx + n]
        idx += n

    # ── console summary ────────────────────────────────────────────────────────
    print(f"\n  t-values (df={df}, t_crit={t_crit:.3f}) — coherence of joints with {target_joint}")
    print(f"  FDR-BH correction across {finite_mask.sum()} tests ({len(joint_order)} joints × {n_freqs} freq bins)")
    print(f"\n  {'Joint':<22} {'Peak t':>8}  {'@ Hz':>7}  {'FDR-sig bins':>13}")
    print("  " + "─" * 56)
    for joint in joint_order:
        t_arr   = t_by_joint[joint]
        sig_arr = reject_by_joint[joint]
        peak_idx = int(np.nanargmax(t_arr))
        n_sig    = int(np.sum(sig_arr))
        print(f"  {joint:<22} {t_arr[peak_idx]:>8.3f}  {freqs[peak_idx]:>7.3f}  {n_sig:>13}")

    if generate_figure:
        # ── build plot_dict ────────────────────────────────────────────────────
        plot_dict = {}
        for joint in joint_order:
            t_arr   = t_by_joint[joint]
            sig_arr = reject_by_joint[joint]
            g = Graph()
            g.add_plot(freqs, t_arr, None, 1)
            plot_dict[joint] = g

        # Significance markers: FDR-surviving bins per joint
        if gesture_band is not None:
            band_lo, band_hi = gesture_band
            band_mask = (freqs >= band_lo) & (freqs <= band_hi)
            signif_x = {j: [(freqs[fi], t_by_joint[j][fi]) for fi in range(n_freqs)
                            if reject_by_joint[j][fi] and band_mask[fi]]
                        for j in joint_order if (reject_by_joint[j] & band_mask).any()}
        else:
            signif_x = {j: [(freqs[fi], t_by_joint[j][fi]) for fi in range(n_freqs)
                            if reject_by_joint[j][fi]]
                        for j in joint_order if reject_by_joint[j].any()}

        path_png = op.join(PATH_FIGURES,
                           f"fig{str(n).zfill(2)}_joint_coherence_ttest_body_graph_"
                           f"{target_joint}{trailing_file_name}_{value_type}.png")

        plot_body_graphs(
            plot_dict,
            title=(f"Coherence with {target_joint} — {sequence_measure} t-values\n"
                   f"(N={n_subjects}, df={df}, FDR-BH α={alpha})"),
            sns_kwargs={"style": "whitegrid",
                        "rc": {"font.family": "Metropolis", "axes.spines.right": False,
                               "axes.spines.top": False}},
            mpl_kwargs={"hspace": 0.5, "tick_pad": 1},
            xlim=[0, 4],
            joint_layout=[[None, None, "Head", None, None],
                          [None, None, "Neck", None, None],
                          ["ElbowRight", "ShoulderRight", "SpineShoulder", "ShoulderLeft", "ElbowLeft"],
                          ["WristRight", None, "SpineMid", None, "WristLeft"],
                          ["HandRight", "HipRight", "SpineBase", "HipLeft", "HandLeft"]],
            overlay_lines=t_crit,
            overlay_lines_color="red",
            overlay_lines_style="--",
            overlay_lines_width=1.5,
            signif_marker="*" if signif_x else None,
            signif_marker_values=[(t_crit, np.inf)] if signif_x else None,
            signif_marker_x_positions=signif_x if signif_x else None,
            signif_marker_color="red",
            signif_marker_size=8,
            show=True,
            path_save=path_png,
        )
        print(f"\n  Saved body graph: {path_png}")

def joint_coherence_lmm(n, target_joint, sequence_measure, value_type="raw", level="subject",
                        freq_max=4.0, alpha=0.05):
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    import statsmodels.formula.api as smf
    import pandas as pd
    import itertools

    trailing_file_name = REREF_SUFFIX if USE_REREF else ""

    print_section_title(f"{n}. JOINT COHERENCE LMM — {target_joint.upper()}")

    if level != "subject":
        print("\n  This analysis requires per-subject variability (subject as a random effect) "
              "and is not meaningful at the group level — pass level='subject'.")
        return

    results_by_subject = _load_coherence_results(sequence_measure, target_joint, level="subject")
    if not results_by_subject:
        return

    subject_names = list(results_by_subject.keys())
    n_subjects = len(subject_names)
    print(f"\n  Found {n_subjects} subject(s): {subject_names}")
    print(f"  Frequency range: 0 < f ≤ {freq_max} Hz  (0 Hz excluded for log transform)")

    # ── load data → long-format dataframe ─────────────────────────────────────
    # One row per subject × joint × frequency bin.
    # value_type="raw"     → Fisher-z transform of raw coherence (column stays comparable to coherence via tanh)
    # value_type="z-score" → permutation z-scores used directly (already null-centred, no tanh back-transform)
    rows = []

    for subject, out in results_by_subject.items():
        data_by_joint, _, lag, freqs = _extract_by_joint(out, target_joint, sequence_measure, value_type)

        freq_mask = (freqs > 0) & (freqs <= freq_max)
        freqs_sel = freqs[freq_mask]

        for joint in UPPER_BODY_JOINTS:
            if joint == target_joint or joint not in data_by_joint:
                continue
            arr = data_by_joint[joint][lag]
            if np.all(np.isnan(arr)):
                continue
            if value_type == "raw":
                value_arr = np.arctanh(np.clip(arr[freq_mask], 0, 1 - 1e-8))
            else:
                value_arr = arr[freq_mask]

            for freq, val in zip(freqs_sel, value_arr):
                rows.append({"subject": subject, "joint": joint,
                             "frequency": freq, "log_freq": np.log(freq),
                             "value": val})

    df_long = pd.DataFrame(rows)
    joint_order = sorted(df_long["joint"].unique())
    n_joints    = len(joint_order)
    n_bins      = df_long["frequency"].nunique()
    log_freq_mean = df_long["log_freq"].mean()   # anchor for slope interpretation
    df_long["log_freq_c"] = df_long["log_freq"] - log_freq_mean   # centred
    df_long["log_freq_c2"] = df_long["log_freq_c"] ** 2

    print(f"\n  {n_joints} joints × {n_bins} frequency bins × {n_subjects} subjects "
          f"= {len(df_long)} observations")
    print(f"  log_freq centred at {log_freq_mean:.3f} (≈ {np.exp(log_freq_mean):.2f} Hz)")

    # ── descriptive table ──────────────────────────────────────────────────────
    # Per-subject OLS slope of value on log_freq for each joint
    subject_slopes = {}   # joint → array of per-subject slopes
    for joint in joint_order:
        slopes = []
        for subject in subject_names:
            df_js = df_long[(df_long["joint"] == joint) & (df_long["subject"] == subject)]
            if len(df_js) < 2:
                slopes.append(np.nan)
                continue
            slope, _, _, _, _ = stats.linregress(df_js["log_freq"], df_js["value"])
            slopes.append(slope)
        subject_slopes[joint] = np.array(slopes)

    value_col_label = "Mean coh" if value_type == "raw" else "Mean z"
    print(f"\n  Descriptive — mean {'coherence' if value_type == 'raw' else 'z-score'} "
          f"and log-frequency slope per joint:")
    print(f"  (intercept at geometric mean freq ≈ {np.exp(log_freq_mean):.2f} Hz)")
    print(f"\n  {'Joint':<22}  {value_col_label:>10}  {'Mean slope':>12}  {'Direction':>10}")
    print("  " + "─" * 60)

    for joint in sorted(joint_order,
                        key=lambda j: -df_long[df_long["joint"] == j]["value"].mean()):
        mean_val   = df_long[df_long["joint"] == joint]["value"].mean()
        display_val = float(np.tanh(mean_val)) if value_type == "raw" else float(mean_val)
        mean_slope = float(np.nanmean(subject_slopes[joint]))
        direction  = "↓ decay" if mean_slope < 0 else "↑ rise"
        print(f"  {joint:<22}  {display_val:>10.4f}  {mean_slope:>12.4f}  {direction:>10}")

    # ── LMM fitting and LRTs ──────────────────────────────────────────────────
    print(f"\n  Fitting LMMs (REML=False for LRT)...")

    def fit(formula):
        model = smf.mixedlm(formula, df_long, groups=df_long["subject"])
        try:
            return model.fit(reml=False, disp=False)
        except Exception:
            return model.fit(reml=False, disp=False, method="nm")

    def lrt(m_full, m_reduced, label):
        lr = 2 * (m_full.llf - m_reduced.llf)
        dof = int(round(m_full.df_modelwc - m_reduced.df_modelwc))
        p = float(stats.chi2.sf(lr, dof))
        sig = "*" if p < alpha else "ns"
        print(f"  {label:<50}  χ²({dof:>3}) = {lr:8.3f},  p = {p:.4f}  {sig}")
        return p

    m_linear = fit("value ~ C(joint) + log_freq_c + C(joint):log_freq_c")
    m_quad_main = fit("value ~ C(joint) + log_freq_c + log_freq_c2 + C(joint):log_freq_c")
    m_quad_full = fit("value ~ C(joint) + log_freq_c + log_freq_c2 + "
                      "C(joint):log_freq_c + C(joint):log_freq_c2")
    m_freq_only = fit("value ~ log_freq_c")
    m_joint_only = fit("value ~ C(joint)")
    m_no_interact = fit("value ~ C(joint) + log_freq_c + log_freq_c2")

    print(f"\n  Likelihood ratio tests (α = {alpha}):")
    print(f"  {'Effect':<50}  {'Result'}")
    print("  " + "─" * 80)
    lrt(m_no_interact, m_freq_only, "Joint main effect")
    lrt(m_no_interact, m_joint_only, "log(frequency) main effect")
    lrt(m_quad_main, m_linear, "Quadratic term (overall spectrum shape)")
    lrt(m_quad_full, m_quad_main, "Joint × quadratic interaction (joint-specific shape)")
    lrt(m_quad_full, m_linear, "Combined quadratic effect (main + interaction)")

    # ── per-joint intercepts and slopes from full model ────────────────────────
    params = m_quad_full.params
    ref_joint     = joint_order[0]   # statsmodels alphabetical reference
    ref_intercept = params["Intercept"]
    ref_slope     = params["log_freq_c"]

    # print(f"\n  Per-joint model estimates (reference joint: {ref_joint}):")
    # print(f"  {'Joint':<22}  {'Intercept':>12}  {'Slope':>10}  {'Coh @ anchor':>14}")
    # print("  " + "─" * 64)

    joint_intercepts = {}
    joint_slopes_model = {}
    for joint in joint_order:
        # if joint == ref_joint:
        #     intercept = ref_intercept
        #     slope     = ref_slope
        # else:
        #     int_key   = f"C(joint)[T.{joint}]"
        #     slp_key   = f"C(joint)[T.{joint}]:log_freq_c"
        #     intercept = ref_intercept + params.get(int_key, 0.0)
        #     slope     = ref_slope     + params.get(slp_key, 0.0)
        # joint_intercepts[joint]    = intercept
        # joint_slopes_model[joint]  = slope
        # coh_anchor = float(np.tanh(intercept))
        # print(f"  {joint:<22}  {intercept:>12.4f}  {slope:>10.4f}  {coh_anchor:>14.4f}")

        # ── per-joint intercepts, linear slopes, quadratic coefficients ───────────
        params = m_quad_full.params
        ref_joint = joint_order[0]

        ref_b0 = params["Intercept"]
        ref_b1 = params["log_freq_c"]
        ref_b2 = params["log_freq_c2"]

        log_freq_c_min = df_long["log_freq_c"].min()
        log_freq_c_max = df_long["log_freq_c"].max()

        print(f"\n  Per-joint quadratic model estimates (reference: {ref_joint}):")
        print(f"  {'Joint':<22}  {'Intercept':>10}  {'Linear b1':>10}  {'Quadratic b2':>13}  "
              f"  {'Peak freq':>10}  {'Shape':>10}")
        print("  " + "─" * 85)

        joint_peaks = {}
        for joint in joint_order:
            if joint == ref_joint:
                b0, b1, b2 = ref_b0, ref_b1, ref_b2
            else:
                b0 = ref_b0 + params.get(f"C(joint)[T.{joint}]", 0.0)
                b1 = ref_b1 + params.get(f"C(joint)[T.{joint}]:log_freq_c", 0.0)
                b2 = ref_b2 + params.get(f"C(joint)[T.{joint}]:log_freq_c2", 0.0)

            MIN_B2 = 1e-3  # below this, curvature is negligible → treat as monotone
            if abs(b2) < MIN_B2:
                peak_freq, in_range = np.nan, False
            elif b2 != 0:
                peak_log_freq_c = -b1 / (2 * b2)
                peak_freq = float(np.exp(peak_log_freq_c + log_freq_mean))
                in_range = log_freq_c_min <= peak_log_freq_c <= log_freq_c_max
            else:
                peak_freq, in_range = np.nan, False

            # Vertex of the parabola in centred log-freq space
            if b2 != 0:
                peak_log_freq_c = -b1 / (2 * b2)
                peak_freq = float(np.exp(peak_log_freq_c + log_freq_mean))
                in_range = log_freq_c_min <= peak_log_freq_c <= log_freq_c_max
            else:
                peak_freq, in_range = np.nan, False

            if b2 < 0 and in_range:
                shape = "peak"
                joint_peaks[joint] = peak_freq
            elif b2 > 0 and in_range:
                shape = "trough"
                joint_peaks[joint] = np.nan
            else:
                shape = "monotone"
                joint_peaks[joint] = np.nan

            peak_str = f"{peak_freq:.3f} Hz" if not np.isnan(peak_freq) else "—"
            print(f"  {joint:<22}  {b0:>10.4f}  {b1:>10.4f}  {b2:>13.4f}  "
                  f"  {peak_str:>10}  {shape:>10}")

        # Summary of joints with a peak within range
        peaked = {j: f for j, f in joint_peaks.items() if not np.isnan(f)}
        if peaked:
            print(f"\n  Joints with a coherence peak within 0–{freq_max} Hz:")
            for joint, freq in sorted(peaked.items(), key=lambda x: x[1]):
                print(f"  {joint:<22}  peak @ {freq:.3f} Hz")
        else:
            print(f"\n  No joints show a coherence peak within 0–{freq_max} Hz "
                  f"(all spectra are monotonically decaying).")

    # ── post-hoc: joint intercepts (overall coherence/z-score level) ──────────
    level_desc = "coherence level" if value_type == "raw" else "z-score level"
    diff_label = "Δcoh" if value_type == "raw" else "Δz"
    print(f"\n  Post-hoc pairwise — joint {level_desc} (marginal means, FDR-BH):")

    joint_marginal = df_long.groupby(["subject", "joint"])["value"].mean().reset_index()
    pairs   = list(itertools.combinations(joint_order, 2))
    pt, pp, pdiff = [], [], []

    for j1, j2 in pairs:
        v1 = joint_marginal[joint_marginal["joint"] == j1]["value"].values
        v2 = joint_marginal[joint_marginal["joint"] == j2]["value"].values
        t, p = stats.ttest_rel(v1, v2)
        pt.append(t); pp.append(p)
        if value_type == "raw":
            pdiff.append(float(np.tanh(v1.mean()) - np.tanh(v2.mean())))
        else:
            pdiff.append(float(v1.mean() - v2.mean()))

    reject, p_corr, _, _ = multipletests(pp, alpha=alpha, method="fdr_bh")
    n_sig = int(reject.sum())
    print(f"  {len(pairs)} pairs tested, {n_sig} significant after FDR-BH\n")

    if n_sig > 0:
        print(f"  {'Joint 1':<22}  {'Joint 2':<22}  {diff_label:>8}  {'t':>7}  {'p_corr':>8}")
        print("  " + "─" * 76)
        for i in np.argsort(pp):
            if not reject[i]:
                continue
            j1, j2 = pairs[i]
            print(f"  {j1:<22}  {j2:<22}  {pdiff[i]:>8.4f}  {pt[i]:>7.3f}  {p_corr[i]:>8.4f}")

    # ── post-hoc: joint slopes (frequency decay rate) ─────────────────────────
    print(f"\n  Post-hoc pairwise — joint frequency decay slope (FDR-BH):")

    slope_pairs = list(itertools.combinations(joint_order, 2))
    st, sp, sdiff = [], [], []

    for j1, j2 in slope_pairs:
        s1 = subject_slopes[j1]
        s2 = subject_slopes[j2]
        t, p = stats.ttest_rel(s1, s2)
        st.append(t); sp.append(p)
        sdiff.append(float(np.nanmean(s1) - np.nanmean(s2)))

    s_reject, sp_corr, _, _ = multipletests(sp, alpha=alpha, method="fdr_bh")
    n_sig_slope = int(s_reject.sum())
    print(f"  {len(slope_pairs)} pairs tested, {n_sig_slope} significant after FDR-BH\n")

    if n_sig_slope > 0:
        print(f"  {'Joint 1':<22}  {'Joint 2':<22}  {'Δslope':>9}  {'t':>7}  {'p_corr':>8}")
        print("  " + "─" * 78)
        for i in np.argsort(sp):
            if not s_reject[i]:
                continue
            j1, j2 = slope_pairs[i]
            print(f"  {j1:<22}  {j2:<22}  {sdiff[i]:>9.4f}  {st[i]:>7.3f}  {sp_corr[i]:>8.4f}")

    return df_long

def joint_coherence_lmm_plots(n, df_long, target_joint, sequence_measure, value_type="raw", generate_figure=True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy import stats

    trailing_file_name = REREF_SUFFIX if USE_REREF else ""

    print_section_title(f"{n}. JOINT COHERENCE LMM PLOTS — {target_joint.upper()}")

    subject_names = sorted(df_long["subject"].unique())
    joint_order   = sorted(df_long["joint"].unique())

    scale_label = "coherence" if value_type == "raw" else "z-score"
    scale_title = "Coherence" if value_type == "raw" else "Z-score"

    # ── compute per-joint summary stats ───────────────────────────────────────
    mean_coh   = {}   # joint → mean value (back-transformed to coherence if value_type="raw")
    mean_slope = {}   # joint → mean OLS slope (log_freq)
    sd_coh     = {}
    sd_slope   = {}

    for joint in joint_order:
        df_j = df_long[df_long["joint"] == joint]

        # mean value per subject (average across freq bins)
        subj_means = df_j.groupby("subject")["value"].mean().values
        if value_type == "raw":
            mean_coh[joint] = float(np.tanh(subj_means.mean()))
            sd_coh[joint]   = float(np.tanh(subj_means.std()))
        else:
            mean_coh[joint] = float(subj_means.mean())
            sd_coh[joint]   = float(subj_means.std())

        # per-subject OLS slopes
        slopes = []
        for subject in subject_names:
            df_js = df_j[df_j["subject"] == subject]
            if len(df_js) < 2:
                continue
            slope, *_ = stats.linregress(df_js["log_freq"], df_js["value"])
            slopes.append(slope)
        slopes = np.array(slopes)
        mean_slope[joint] = float(slopes.mean())
        sd_slope[joint]   = float(slopes.std())

    # ── silhouette 1: mean coherence / z-score ────────────────────────────────
    val_dict_coh = {j: mean_coh.get(j, 0.0) for j in joints}
    val_dict_coh[target_joint] = 0.0
    val_max_coh = max(v for j, v in val_dict_coh.items() if j != target_joint)

    path_sil_coh = op.join(PATH_FIGURES,
                           f"fig{str(n).zfill(2)}_lmm_silhouette_{scale_label}_"
                           f"{target_joint}{trailing_file_name}.png")
    plot_silhouette(
        val_dict_coh,
        title=f"Coherence with {target_joint}\nmean {scale_label} (0–4 Hz)",
        title_scale=scale_title,
        min_scale=0, max_scale=val_max_coh,
        color_scheme="celsius",
        show_values=True, value_format=".2f", strip_leading_zero=True,
        font_size=0.8, show=False,
        path_save=path_sil_coh,
        resolution=(800, 1080), verbosity=0,
    )
    print(f"  Saved: {path_sil_coh}")

    # ── silhouette 2: decay slope (absolute value → steeper = higher) ─────────
    val_dict_slope = {j: abs(mean_slope.get(j, 0.0)) for j in joints}
    val_dict_slope[target_joint] = 0.0
    val_max_slope = max(v for j, v in val_dict_slope.items() if j != target_joint)

    path_sil_slope = op.join(PATH_FIGURES,
                             f"fig{str(n).zfill(2)}_lmm_silhouette_slope_{scale_label}_"
                             f"{target_joint}{trailing_file_name}.png")
    plot_silhouette(
        val_dict_slope,
        title=f"Coherence with {target_joint}\nfrequency decay rate |slope|",
        title_scale="|Slope|",
        min_scale=0, max_scale=val_max_slope,
        color_scheme="celsius",
        show_values=True, value_format=".2f", strip_leading_zero=True,
        font_size=0.8, show=False,
        path_save=path_sil_slope,
        resolution=(800, 1080), verbosity=0,
    )
    print(f"  Saved: {path_sil_slope}")

    # ── scatter: mean coherence vs decay slope ────────────────────────────────
    path_scatter = op.join(PATH_FIGURES,
                           f"fig{str(n).zfill(2)}_lmm_scatter_{scale_label}_"
                           f"{target_joint}{trailing_file_name}.png")

    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "Metropolis"

    fig, ax = plt.subplots(figsize=(7, 5.5))

    x_vals = np.array([mean_coh[j]          for j in joint_order])
    y_vals = np.array([abs(mean_slope[j])   for j in joint_order])
    x_sd   = np.array([sd_coh[j]            for j in joint_order])
    y_sd   = np.array([sd_slope[j]          for j in joint_order])

    ax.errorbar(x_vals, y_vals, xerr=x_sd, yerr=y_sd,
                fmt="none", color="grey", alpha=0.4, linewidth=0.8, zorder=1)
    ax.scatter(x_vals, y_vals, s=60, color="#2196F3", zorder=2, clip_on=False)

    # joint labels — offset to avoid overlap
    for i, joint in enumerate(joint_order):
        x_off = 0.004
        y_off = 0.003
        ax.annotate(joint, (x_vals[i] + x_off, y_vals[i] + y_off),
                    fontsize=7.5, color="#333333", va="bottom")

    # regression line
    slope_reg, intercept_reg, r, p_val, _ = stats.linregress(x_vals, y_vals)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, intercept_reg + slope_reg * x_line,
            color="#E53935", linewidth=1.2, linestyle="--",
            label=f"r = {r:.2f}, p = {p_val:.3f}")

    ax.set_xlabel(f"Mean {scale_label}", fontsize=10)
    ax.set_ylabel("Frequency decay rate |slope|", fontsize=10)
    ax.set_title(f"Coherence with {target_joint} — {sequence_measure}\n"
                 f"Coupling strength vs. frequency specificity", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    fig.savefig(path_scatter, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {path_scatter}")

def joint_coherence_matrix(n, sequence_measure, frequency=None, value_type="z-score", level="group",
                           generate_figure=True):
    trailing_file_name = REREF_SUFFIX if USE_REREF else ""

    level_label = "GROUP" if level == "group" else "PER SUBJECT (averaged)"
    print_section_title(f"{n}. JOINT COHERENCE MATRIX ({level_label})")

    joints_test = [j for j in joints if j != "SpineBase"] if USE_REREF else joints.copy()
    n_joints = len(joints_test)

    has_significance = (value_type == "z-score" and level == "group")

    # ── Resolve frequency parameter to a scalar extractor ────────────────────
    # frequency: None   → mean value across all freqs
    #            float  → value at closest single frequency
    #            tuple  → mean value within (lo, hi) band

    def _scalar(arr, freqs):
        """Extract scalar value for this pair given the frequency parameter."""
        if np.all(np.isnan(arr)):
            return np.nan
        if frequency is None:
            return float(np.nanmean(arr))
        elif isinstance(frequency, tuple):
            lo, hi = frequency
            mask = (freqs >= lo) & (freqs <= hi)
            return float(np.nanmean(arr[mask]))
        else:
            idx = int(np.argmin(np.abs(freqs - frequency)))
            return float(arr[idx])

    def _threshold_from(z_arr, p_arr, freqs):
        """Max z where p >= alpha within the selected frequency range (boundary z)."""
        if frequency is None:
            mask = np.ones(len(freqs), dtype=bool)
        elif isinstance(frequency, tuple):
            lo, hi = frequency
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            idx = int(np.argmin(np.abs(freqs - frequency)))
            mask = np.zeros(len(freqs), dtype=bool)
            mask[idx] = True
        nonsig = mask & (p_arr >= 0.05)
        if nonsig.any():
            return float(z_arr[nonsig].max())
        return np.nan

    # ── Load all result files and build the value matrix ──────────────────────
    print(f"\nLoading coherence results for {n_joints} joints...")

    z_matrix = pd.DataFrame(np.nan, index=joints_test, columns=joints_test)
    thresh_vals = []  # for deriving group threshold (z-score, group level only)
    out_ref = None

    for target_joint in joints_test:
        if level == "group":
            out = _load_coherence_results(sequence_measure, target_joint, level="group")
            if out is None:
                print(f"  ⚠ Missing group file for {target_joint}")
                continue
            out_ref = out
            data_by_joint, p_by_joint, lag, freqs = _extract_by_joint(out, target_joint, sequence_measure, value_type)

        else:  # level == "subject" — average per-subject values for this target joint
            results_by_subject = _load_coherence_results(sequence_measure, target_joint, level="subject")
            if not results_by_subject:
                print(f"  ⚠ No subject files for {target_joint}")
                continue
            stacked = {}
            for out in results_by_subject.values():
                out_ref = out
                subj_data_by_joint, _, lag, freqs = _extract_by_joint(out, target_joint, sequence_measure, value_type)
                for other_joint, arr_by_lag in subj_data_by_joint.items():
                    stacked.setdefault(other_joint, []).append(arr_by_lag[lag])
            data_by_joint = {j: {lag: np.nanmean(arrs, axis=0)} for j, arrs in stacked.items()}
            p_by_joint = None

        for other_joint in joints_test:
            if other_joint not in data_by_joint:
                continue
            arr = data_by_joint[other_joint][lag]
            z_matrix.loc[target_joint, other_joint] = _scalar(arr, freqs)

        # Collect boundary-z values for threshold estimation (only meaningful with per-bin p-values)
        if has_significance:
            for other_joint in joints_test:
                if other_joint == target_joint or other_joint not in data_by_joint:
                    continue
                arr = data_by_joint[other_joint][lag]
                p_arr = p_by_joint[other_joint][lag]
                if np.all(np.isnan(arr)):
                    continue
                t = _threshold_from(arr, p_arr, freqs)
                if np.isfinite(t):
                    thresh_vals.append(t)

        print(f"  ✓ {target_joint}")

    # Diagonal is meaningless (self-coherence) — set to NaN
    for j in joints_test:
        z_matrix.loc[j, j] = np.nan

    # ── Significance threshold (z-score, group level only) ────────────────────
    if has_significance:
        z_threshold = float(np.median(thresh_vals)) if thresh_vals else 1.96
        sig_matrix = z_matrix > z_threshold
    else:
        z_threshold = None
        sig_matrix = pd.DataFrame(True, index=joints_test, columns=joints_test)  # no hatching

    value_label = "z-score" if value_type == "z-score" else "coherence"

    # ── Print summary ─────────────────────────────────────────────────────────
    freq_label = (
        "all frequencies" if frequency is None
        else f"{frequency[0]}–{frequency[1]} Hz" if isinstance(frequency, tuple)
        else f"{frequency} Hz"
    )
    print(f"\nFrequency: {freq_label}")
    if has_significance:
        print(f"Significance threshold z = {z_threshold:.3f}  "
              f"(median boundary, α = {out_ref.signif_alpha}, "
              f"n_perms = {out_ref.number_of_randperms})")
        n_sig = int(sig_matrix.values[~np.eye(n_joints, dtype=bool)].sum()) // 2
        n_total = n_joints * (n_joints - 1) // 2
        print(f"Significant pairs: {n_sig}/{n_total}")
    else:
        print(f"No per-bin significance available for value_type={value_type!r}, level={level!r} "
              f"— showing {value_label} values without significance masking.")

    print(f"\n  {'Joint':<22} {f'Mean {value_label} (row)':>18}  {'Sig pairs':>10}")
    print("  " + "─" * 54)
    for j in joints_test:
        row = z_matrix.loc[j].drop(index=j)
        mean_val = float(np.nanmean(row))
        n_sig_j = int(sig_matrix.loc[j].drop(index=j).sum()) if has_significance else 0
        print(f"  {j:<22} {mean_val:>18.2f}  {n_sig_j:>10}")

    if not generate_figure:
        return

    # ── Build anatomical ordering from JOINT_GROUPS ───────────────────────────
    ordered = []
    for group_joints in JOINT_GROUPS.values():
        ordered.extend([j for j in group_joints if j in joints_test])
    ordered.extend([j for j in joints_test if j not in ordered])

    z_ord = z_matrix.reindex(index=ordered, columns=ordered)
    sig_ord = sig_matrix.reindex(index=ordered, columns=ordered)
    n_ord = len(ordered)

    # Group boundaries and label positions
    group_boundaries = []
    group_label_positions = {}
    pos = 0
    for group_name, group_joints in JOINT_GROUPS.items():
        valid = [j for j in group_joints if j in joints_test]
        if not valid:
            continue
        group_label_positions[group_name] = pos + len(valid) / 2 - 0.5
        pos += len(valid)
        group_boundaries.append(pos)
    group_boundaries = group_boundaries[:-1]

    # ── Heatmap ───────────────────────────────────────────────────────────────
    cell_size = 0.55
    fig_size = n_ord * cell_size + 3.5
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    cmap = LinearSegmentedColormap.from_list(
        "celsius",
        ["#40E0D0",  # turquoise
         "#8FD400",  # sheen green
         "#FFCC00",  # tangerine yellow
         "#FF8C00",  # orange
         "#FF0000"],  # red
        N=256
    )

    # Cap vmax at 99th percentile of off-diagonal to prevent self-coherence dominance
    off_diag = z_ord.values[~np.eye(n_ord, dtype=bool)]
    vmax = float(np.nanpercentile(off_diag, 99))
    vmax = max(vmax, 1.0)

    sns.heatmap(
        z_ord,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": f"{value_label.capitalize()}  ({freq_label})", "shrink": 0.6},
        xticklabels=True,
        yticklabels=True,
        annot=False,
    )

    for row_i, ji in enumerate(ordered):
        for col_i, jj in enumerate(ordered):
            z_val = z_ord.loc[ji, jj]
            is_diag = (ji == jj)
            is_sig = sig_ord.loc[ji, jj]

            if is_diag:
                ax.add_patch(plt.Rectangle((col_i, row_i), 1, 1,
                                           fill=True, facecolor="#dddddd", zorder=3))
                ax.text(col_i + 0.5, row_i + 0.5, "—",
                        ha="center", va="center", fontsize=7, color="#888")
            else:
                text_color = "white" if z_val > vmax * 0.6 else "black"
                ax.text(col_i + 0.5, row_i + 0.5, f"{z_val:.1f}",
                        ha="center", va="center", fontsize=6.5, color=text_color)
                if not is_sig:
                    ax.add_patch(plt.Rectangle((col_i, row_i), 1, 1,
                                               fill=False, hatch="////",
                                               edgecolor="black", linewidth=0,
                                               alpha=0.25, zorder=3))

    for boundary in group_boundaries:
        ax.axhline(boundary, color="black", linewidth=1.8, zorder=4)
        ax.axvline(boundary, color="black", linewidth=1.8, zorder=4)

    for group_name, mid in group_label_positions.items():
        ax.text((mid + 0.5) / n_ord, 1.012, group_name,
                ha="center", va="bottom", fontsize=8,
                fontstyle="italic", color="#333",
                transform=ax.transAxes)
        ax.text(-0.01, 1.0 - (mid + 0.5) / n_ord, group_name,
                ha="right", va="center", fontsize=8,
                fontstyle="italic", color="#333",
                rotation=90, transform=ax.transAxes)

    subtitle = (f"{value_label.capitalize()}s (permutation null)  |  Hatched = z ≤ {z_threshold:.2f} (non-significant)"
                if has_significance else f"{value_label.capitalize()}s ({level})")
    ax.set_title(
        f"Joint-to-joint coherence matrix  ({freq_label})\n{subtitle}",
        fontsize=11, pad=28
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)
    save_figure(fig, PATH_FIGURES,
                f"fig{str(n).zfill(2)}_joint_coherence_matrix_{sequence_measure}"
                f"{trailing_file_name}_{value_type}_{level}.png",
                **SAVE_KW)
    plt.close(fig)


def power_spectrum_analysis(n, df, sequence_measure, generate_figure=True, **kwargs):
    from krajjat import analysis_functions as afunc

    if USE_REREF:
        trailing_file_name = "_rSpineBase"
    else:
        trailing_file_name = ""

    print_section_title(f"{n}. POWER SPECTRUM — {sequence_measure.upper()}")

    path_png = op.join(PATH_FIGURES,
                       f"fig{str(n).zfill(2)}_power_spectrum_{sequence_measure}"
                       f"{trailing_file_name}.png") if generate_figure else None
    path_json = op.join(PATH_RESULTS, "PowerSpectrum",
                        f"power_spectrum_{sequence_measure}{trailing_file_name}.json")

    if USE_REREF:
        df_ps = df[df["label"] != "SpineBase"].copy()
    else:
        df_ps = df

    # Defaults — all overridable via kwargs
    params = dict(
        method="welch",
        sequence_measure=sequence_measure,
        audio_measure=None,
        subjects=subjects,
        average=None,
        series="subject",
        result_type="average",
        permutation_method=None,
        number_of_randperms=0,
        fit_background=True,
        background_lower_freq=0.5,
        signif_style=["shade", "markers"],
        signif_alpha=0.05,
        signif_tail="1",
        signif_direction="up",
        joint_layout="kinect",
        n_jobs=-1,
        random_seed=42,
        show=False,
        path_save=path_png,
        verbosity=1,
    )
    params.update(kwargs)

    out = afunc.power_spectrum(df_ps, **params)
    out.save(path_json)

def _load_psd_file(kinematic_measure, acoustic_measure, use_reref=True, use_filtered=True, file_trail=""):
    # Load data
    print("Loading dataframe...")
    if use_filtered:
        f = "filtered"
    else:
        f = "unfiltered"

    if use_reref:
        path = op.join(PATH_DATAFRAME, f"FS{int(FS)} {f}", f"kinect_{kinematic_measure}_{acoustic_measure}_rSpineBase{file_trail}.gzip")
    else:
        path = op.join(PATH_DATAFRAME, f"FS{int(FS)} {f}", f"kinect_{kinematic_measure}_{acoustic_measure}{file_trail}.gzip")
    df = pd.read_parquet(path)

    return df

def _filter_df(df, subjects, trials, joints):
    if subjects is not None:
        subjects = [subjects] if isinstance(subjects, str) else subjects
        df = df[df["subject"].isin(subjects)]
    if trials is not None:
        trials = [trials] if isinstance(trials, str) else trials
        df = df[df["trial"].isin(trials)]
    if joints is not None:
        joints = [joints] if isinstance(joints, str) else joints
        df = df[df["label"].isin(joints)]
    unique_subjects = df["subject"].unique()
    unique_trials = df["trial"].unique()
    unique_joints = df["label"].unique()
    print(f"\t{len(unique_subjects)} subject(s), {len(unique_trials)} trial(s), {len(unique_joints)} joint(s)")
    print(f"Dataframe length: {len(df)} rows")
    return df

def _infer_fs(df):
    subj = df["subject"].iloc[0]
    trial = df["trial"].iloc[0]
    label = df["label"].iloc[0]
    group = df[(df["subject"] == subj) & (df["trial"] == trial) & (df["label"] == label)]
    timestamps = group.sort_values("timestamp")["timestamp"].values
    return 1 / np.median(np.diff(timestamps))

def _normalize(mean_psd, sd_psd, freqs, normalization, global_max, subject_max, band_max_range=(0.5, 4.0)):
    if normalization is None:
        factor = 1.0
    elif normalization == "*f":
        factor = 1 / freqs  # element-wise
    elif normalization == "subject_peak":
        factor = subject_max
    elif normalization == "global_peak":
        factor = global_max
    elif normalization == "total_power":
        total = np.trapezoid(mean_psd, freqs)
        factor = total if total > 0 else 1.0
    elif normalization == "band_maximum":
        band_mask = (freqs >= band_max_range[0]) & (freqs <= band_max_range[1])
        band_max_val = mean_psd[band_mask].max() if band_mask.any() else mean_psd.max()
        factor = band_max_val if band_max_val > 0 else 1.0
    elif normalization == "fooof":
        from specparam import SpectralModel
        from specparam.sim.gen import gen_aperiodic
        fm = SpectralModel(verbose=False)
        fm.fit(freqs[1:], np.maximum(mean_psd[1:], 1e-30), freq_range=[1, 10])
        ap_params = fm.results.params.aperiodic.params
        ap_fit = gen_aperiodic(freqs[1:], fm.modes.aperiodic, ap_params)
        factor = np.concatenate([[1.0], 10 ** ap_fit])

        raw_peak_freq = freqs[1:][np.argmax(mean_psd[1:])]
        fooof_peak_freq = freqs[1:][np.argmax(mean_psd[1:] / (10 ** ap_fit))]
        print(f"    Raw peak: {raw_peak_freq:.2f} Hz — FOOOF peak: {fooof_peak_freq:.2f} Hz")
    else:
        raise ValueError(f"Wrong value for the parameter normalization: {normalization}.")

    return mean_psd / factor, (sd_psd / factor if sd_psd is not None else None)


def _get_psds(df, fs, nperseg, normalization, band_max_range, psd_unit):
    subj_list = sorted(df["subject"].unique())
    joint_list = sorted(df["label"].unique())

    psds = {}
    freqs = None

    for joint in joint_list:
        psds[joint] = {}

        for subj in subj_list:
            rows = df[(df["subject"] == subj) & (df["label"] == joint)]
            if rows.empty:
                continue
            trial_psds = []
            for trial in sorted(rows["trial"].unique()):
                values = rows[rows["trial"] == trial].sort_values("timestamp")["value"].to_numpy()
                if len(values) < nperseg:
                    continue
                freqs, psd = welch(values, fs=fs, nperseg=nperseg)
                trial_psds.append(psd)
            if not trial_psds:
                continue
            mean_psd = np.mean(trial_psds, axis=0)
            sd_psd = np.std(trial_psds, axis=0) if len(trial_psds) > 1 else None
            psds[joint][subj] = (mean_psd, sd_psd)

    # Global max across all joints and subjects
    global_max = np.max([psds[joint][subj][0].max()
                         for joint in psds if joint != "Audio"
                         for subj in psds[joint]])

    subjects_max = {subj: np.max([psds[joint][subj][0].max()
                                  for joint in psds
                                  if subj in psds[joint] and joint != "Audio"])
                    for subj in subj_list}

    for joint in joint_list:
        for subj in psds[joint]:
            mean_psd, sd_psd = psds[joint][subj]
            subject_max = subjects_max[subj]
            psds[joint][subj] = _normalize(mean_psd, sd_psd, freqs, normalization, global_max, subject_max, band_max_range)

            # Inside the for subj loop, after _normalize:
            if psd_unit == "dB":
                mean_psd, sd_psd = psds[joint][subj]
                psds[joint][subj] = (10 * np.log10(mean_psd + 1e-30),
                                     10 * np.log10(sd_psd + 1e-30) if sd_psd is not None else None)

    return psds, freqs


def _stats_joints(psds, freqs, path_png_silhouette, show):
    from scipy.stats import friedmanchisquare

    print_section_title("STATS — JOINT POWER")

    joint_list = [j for j in psds if j != "Audio" and not (USE_REREF and j == "SpineBase")]
    subj_list = sorted(set.intersection(*[set(psds[j].keys()) for j in joint_list]))
    n, k = len(subj_list), len(joint_list)

    # Raw band power per joint per subject
    band_power = {j: {s: np.trapezoid(psds[j][s][0], freqs) for s in subj_list} for j in joint_list}

    # Percentage of total body power per subject, then averaged
    pct_power = {}
    for s in subj_list:
        total = sum(band_power[j][s] for j in joint_list)
        for j in joint_list:
            if j not in pct_power:
                pct_power[j] = {}
            pct_power[j][s] = 100 * band_power[j][s] / total if total > 0 else 0.0

    mean_pct = {j: np.mean(list(pct_power[j].values())) for j in joint_list}
    sd_pct   = {j: np.std(list(pct_power[j].values())) for j in joint_list}

    print("Joint ordering by percentage of total body power (M ± SD):")
    for rank, j in enumerate(sorted(mean_pct, key=mean_pct.get, reverse=True), 1):
        print(f"  {rank:2d}. {j}: {mean_pct[j]:.1f}% ± {sd_pct[j]:.1f}%")

    # Friedman test on raw band power
    matrix = np.array([[band_power[j][s] for j in joint_list] for s in subj_list])
    stat, p = friedmanchisquare(*matrix.T)
    eta2 = stat / (n * (k - 1))
    print(f"\nFriedman test: χ²({k - 1}) = {stat:.3f}, p = {p:.3f}, η² = {eta2:.3f}")

    silhouette_dict = {j: mean_pct[j] for j in joint_list}
    plot_silhouette(silhouette_dict,
                    show_scale=True, title_scale="% of total power",
                    color_scheme="celsius",
                    show_values=True, value_format=".1f",
                    # title="Band power distribution (0–4 Hz)",
                    path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"),
                    show=show,
                    path_save=path_png_silhouette,
                    resolution=(600, 1080),
                    min_scale=0,
                    font_size=0.8,
                    verbosity=0)

def _stats_subjects(psds, freqs, z_threshold=2.0, freq_bands=None):
    print_section_title("STATS — SUBJECT VARIABILITY")

    joint_list = [j for j in psds if j != "Audio" and not (USE_REREF and j == "SpineBase")]
    subj_list = sorted(set.intersection(*[set(psds[j].keys()) for j in joint_list]))

    if freq_bands is None:
        freq_bands = {"0–1 Hz": (0, 1), "1–2 Hz": (1, 2), "2–4 Hz": (2, 4)}

    # ── CV per joint ──────────────────────────────────────────────────────────
    band_power = {j: {s: np.trapezoid(psds[j][s][0], freqs) for s in subj_list} for j in joint_list}
    print("Coefficient of variation (SD/mean) of total power per joint:")
    cv_per_joint = {}
    for j in joint_list:
        values = np.array([band_power[j][s] for s in subj_list])
        cv = values.std() / values.mean() if values.mean() > 0 else np.nan
        cv_per_joint[j] = cv
    for j in sorted(cv_per_joint, key=lambda x: cv_per_joint[x] if not np.isnan(cv_per_joint[x]) else -1,
                    reverse=True):
        print(f"  {j}: CV = {cv_per_joint[j]:.3f}" if not np.isnan(cv_per_joint[j]) else f"  {j}: CV = nan")

    # ── Per-frequency-bin z-scores ────────────────────────────────────────────
    print(f"\nSubject deviations from group mean (|z| > {z_threshold}) by joint and frequency band:")
    flagged = []

    for joint in joint_list:
        psd_matrix = np.array([psds[joint][s][0] for s in subj_list])  # (n_subj, n_freqs)
        mean_across = psd_matrix.mean(axis=0)
        sd_across = psd_matrix.std(axis=0)
        z_matrix = np.where(sd_across > 0, (psd_matrix - mean_across) / sd_across, 0)

        for band_name, (f_low, f_high) in freq_bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if not band_mask.any():
                continue
            band_z = z_matrix[:, band_mask].mean(axis=1)  # mean z per subject within band
            for i, subj in enumerate(subj_list):
                if abs(band_z[i]) > z_threshold:
                    flagged.append((subj, joint, band_name, band_z[i]))

    if flagged:
        flagged.sort(key=lambda x: abs(x[3]), reverse=True)
        for subj, joint, band, z in flagged:
            direction = "↑" if z > 0 else "↓"
            print(f"  {direction} {subj} — {joint} — {band}: z = {z:.2f}")
    else:
        print(f"  No deviations found.")


def _get_plot_dictionary(psds, freqs, show_average, freq_log, show_sd, skip_joints=()):
    if freq_log:
        mask = freqs > 0
        freqs = freqs[mask]
        psds = {joint: {subj: (mean[mask], sd[mask] if sd is not None else None)
                        for subj, (mean, sd) in subj_psds.items()}
                for joint, subj_psds in psds.items()}

    plot_dictionary = {}
    for joint, subj_psds in psds.items():
        graph = Graph()

        if joint in skip_joints:
            plot_dictionary[joint] = graph
            continue

        if show_average == "bottom" and subj_psds:
            means = [mean for mean, _ in subj_psds.values()]
            mean_psd = np.nanmean(means, axis=0)
            sd_psd = np.nanstd(means, axis=0) if show_sd and len(means) > 1 else None
            graph.add_plot(freqs, mean_psd, sd_psd, 2.5, "-", "#30303080", "mean")

        for subj, (mean_psd, sd_psd) in subj_psds.items():
            sd = sd_psd if show_sd else None
            graph.add_plot(freqs, mean_psd, sd, 1, "-", SUBJECT_COLORS[subj], subj)

        if show_average == "top" and subj_psds:
            means = [mean for mean, _ in subj_psds.values()]
            mean_psd = np.nanmean(means, axis=0)
            sd_psd = np.nanstd(means, axis=0) if show_sd and len(means) > 1 else None
            graph.add_plot(freqs, mean_psd, sd_psd, 2.5, "-", "#30303080", "mean")

        plot_dictionary[joint] = graph

    return plot_dictionary


def plot_psd(n, kinematic_measure, acoustic_measure, df=None, use_reref=True, use_filtered=True,
             file_trail="", subjects=None, trials=None, joints=None,
             nperseg=256, normalization=None, band_max_range=(0.5, 4.0),
             psd_unit="default", freq_log=False, show_stats=True,
             show_average=True, show_sd=False, show=True, path_png=None, path_png_silhouette=None, **kwargs):

    print_section_title(f"{n}. PSD DIAGNOSTIC — {kinematic_measure.upper()} / {acoustic_measure.upper()}")
    if df is None:
        df = _load_psd_file(kinematic_measure, acoustic_measure, use_reref, use_filtered, file_trail)
    df = _filter_df(df, subjects, trials, joints)
    fs = _infer_fs(df)
    psds, freqs = _get_psds(df, fs, nperseg, normalization, band_max_range, psd_unit)
    if show_stats:
        _stats_joints(psds, freqs, path_png_silhouette, show)
        _stats_subjects(psds, freqs)
    skip = ("SpineBase", ) if use_reref else ()
    plot_dictionary = _get_plot_dictionary(psds, freqs, show_average, freq_log, show_sd, skip_joints=skip)
    fig = plot_body_graphs(plot_dictionary, x_scale_log=freq_log, show=False, path_save=None, **kwargs)
    if use_reref:
        _annotate_reref_axis(fig)
    if path_png is not None:
        fig.savefig(path_png, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    # plt.close(fig)
    # plot_body_graphs(plot_dictionary, x_scale_log=freq_log, show=show, path_save=path_png,
    #                  **kwargs)


def pca_analysis(n, df, sequence_measure, n_components=3, generate_figure=True):
    from krajjat import analysis_functions as afunc

    if USE_REREF:
        df_pca = df[df["label"] != "SpineBase"].copy()
    else:
        df_pca = df.copy()

    print_section_title(f"{n}. PCA — {sequence_measure.upper()} (trial level, per subject)")

    joints_test = [j for j in joints if j != "SpineBase"] if USE_REREF else joints.copy()

    # ── Per-subject PCA ───────────────────────────────────────────────────────
    all_loadings  = {}   # subject → list of component loading dicts
    all_var_ratio = {}   # subject → explained_variance_ratio array

    for subj in subjects:
        subj_short = subj.split("_")[1]
        print(f"\n{subj_short}:")

        df_subj = df_pca[df_pca["subject"] == subj]

        _, pca_comps, var_ratio = afunc.pca(
            df_subj,
            n_components=n_components,
            sequence_measure=sequence_measure,
            audio_measure=None,
            level="trial",
            nan_behaviour="zero",
            random_seed=42,
            show=False,
            path_save=None,
            verbosity=0,
        )

        # Align signs: flip each component so that the joint with the largest
        # absolute mean loading across subjects is always positive.
        # Reference joint per PC = the one with highest |loading| in this subject.
        # We align to HandLeft if present, otherwise to the top-loading joint.
        reference_joint = "HandLeft" if "HandLeft" in pca_comps[0] else None
        aligned_comps = []
        for comp in pca_comps:
            if reference_joint is not None:
                ref_val = comp[reference_joint]
            else:
                ref_val = max(comp.values(), key=abs)
            if ref_val < 0:
                comp = {j: -v for j, v in comp.items()}
            aligned_comps.append(comp)
        pca_comps = aligned_comps

        all_loadings[subj_short]  = pca_comps
        all_var_ratio[subj_short] = var_ratio

        for i, (comp, var) in enumerate(zip(pca_comps, var_ratio)):
            top3 = sorted(comp.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            top3_str = ", ".join(f"{j} ({abs(v):.3f})" for j, v in top3)
            print(f"  PC{i+1}: {var*100:.1f}% variance  |  top joints: {top3_str}")

    # ── Group-level summary: mean loadings across subjects ────────────────────
    print(f"\nGroup-mean explained variance per PC:")
    for i in range(n_components):
        mean_var = np.mean([all_var_ratio[s][i] for s in all_var_ratio
                            if i < len(all_var_ratio[s])])
        print(f"  PC{i+1}: {mean_var*100:.1f}%")

    print(f"\nGroup-mean loadings per PC (averaged across subjects):")
    for i in range(n_components):
        mean_comp = {}
        for j in joints_test:
            vals = [all_loadings[s][i][j] for s in all_loadings
                    if i < len(all_loadings[s]) and j in all_loadings[s][i]]
            if vals:
                mean_comp[j] = float(np.mean(vals))
        top5 = sorted(mean_comp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top5_str = ", ".join(f"{j} ({v:+.3f})" for j, v in top5)
        print(f"  PC{i+1}: {top5_str}")

    if not generate_figure:
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure A — One silhouette grid per PC, one column per subject
    # Loadings can be positive or negative → diverging colormap
    # ═══════════════════════════════════════════════════════════════════════════
    for pc_idx in range(n_components):
        # Build {joint: array_of_loadings} one value per subject
        loading_dict = {}
        for j in joints_test:
            loading_dict[j] = np.array([
                abs(all_loadings[s][pc_idx].get(j, 0.0))
                for s in [subj.split("_")[1] for subj in subjects]
            ])

        # Ensure all joints present
        for j in joints:
            loading_dict.setdefault(j, np.zeros(len(subjects)))

        vmax = max(abs(v) for arr in loading_dict.values() for v in arr)
        vmax = max(vmax, 0.05)

        mean_var = np.mean([all_var_ratio[s][pc_idx]
                            for s in all_var_ratio
                            if pc_idx < len(all_var_ratio[s])]) * 100

        plot_silhouette(
            loading_dict,
            title=f"PC{pc_idx + 1} loadings — {sequence_measure}  "
                  f"(mean explained variance: {mean_var:.1f}%)",
            title_silhouette=[s.split("_")[1] for s in subjects],
            title_scale="|Loading|",
            min_scale=0,
            max_scale=vmax,
            color_scheme="celsius",
            show_values=True,
            value_format=".2f",
            font_size=0.5,
            n_cols=3,
            show=False,
            path_save=op.join(PATH_FIGURES,
                              f"fig{str(n).zfill(2)}a_pca_pc{pc_idx+1}_loadings.png"),
            resolution=(1000, 1080),
            verbosity=0,
        )

        # Group-mean silhouette: average loading per joint across subjects
        mean_loading_dict = {j: float(np.mean(loading_dict[j])) for j in loading_dict}
        vmax_mean = max(abs(v) for v in mean_loading_dict.values())
        vmax_mean = max(vmax_mean, 0.05)

        plot_silhouette(
            mean_loading_dict,
            title=f"PC{pc_idx + 1} — group-mean loadings\n"
                  f"{sequence_measure}  (mean explained variance: {mean_var:.1f}%)",
            title_scale="Mean |loading|",
            min_scale=0,
            max_scale=vmax_mean,
            color_scheme="celsius",
            show_values=True,
            value_format=".2f",
            font_size=0.8,
            show=False,
            path_save=op.join(PATH_FIGURES,
                              f"fig{str(n).zfill(2)}b_pca_pc{pc_idx + 1}_loadings_mean.png"),
            resolution=(800, 1080),
            verbosity=0,
        )

def _make_synthetic_psd_df(fs=20, duration=60, joint="HandRight", measure="velocity", n_trials=3, seed=42):
    np.random.seed(seed)
    t = np.arange(0, duration, 1 / fs)
    n = len(t)

    def pink_noise(n):
        white = np.random.randn(n)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1
        fft /= np.sqrt(freqs)
        return np.real(np.fft.irfft(fft, n))

    subjects = {
        "S1_pure_0.5Hz":  np.sin(2 * np.pi * 0.5 * t),
        "S2_pure_1.5Hz":  np.sin(2 * np.pi * 1.5 * t),
        "S3_dual_peak":   np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 1.66 * t),
        "S4_white_noise": np.random.randn(n),
        "S5_pink_noise":  pink_noise(n),
    }

    rows = []
    for subj, signal in subjects.items():
        for trial_i in range(n_trials):
            noise = np.random.randn(n) * 0.05
            for i, (ts, val) in enumerate(zip(t, signal + noise)):
                rows.append({"subject": subj, "trial": f"T{trial_i + 1:03d}",
                             "label": joint, "measure": measure, "timestamp": ts, "value": val})

    return pd.DataFrame(rows)


def _decode(obj):
    """Recursively decode Krajjat custom JSON types into native Python/NumPy objects."""
    if isinstance(obj, dict):
        t = obj.get("__type__")
        if t == "ndarray":
            return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
        elif t == "dict":
            return {(k if isinstance(k, (str, int, float)) else str(k)): _decode(v)
                    for k, v in obj["items"]}
        elif t == "tuple":
            return tuple(_decode(x) for x in obj["data"])
        else:
            return {k: _decode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode(x) for x in obj]
    return obj


def _find_graph_plots(obj, results=None):
    """
    Recursively search the raw JSON for GraphPlot objects and return their decoded attrs.
    Each GraphPlot represents one condition's coherence spectrum.
    """
    if results is None:
        results = []
    if isinstance(obj, dict):
        if obj.get("__type__") == "object" and obj.get("class") == "GraphPlot":
            attrs_raw = obj.get("attrs", {})
            attrs = {k: _decode(v) for k, v in attrs_raw.get("items", [])}
            results.append(attrs)
        else:
            for v in obj.values():
                _find_graph_plots(v, results)
    elif isinstance(obj, list):
        for item in obj:
            _find_graph_plots(item, results)
    return results


# ── File loading ───────────────────────────────────────────────────────────────

def load_coherence_file(fpath, freq_min=0, freq_max=4):
    """
    Parse a *_conds.json file and return mean coherence per condition
    averaged over [freq_min, freq_max] Hz.

    Returns
    -------
    dict[str, float]  e.g. {"Masked": 0.012, "Unmasked": 0.009}
    """
    with open(fpath) as f:
        raw = json.load(f)

    graph_plots = _find_graph_plots(raw)

    result = {}
    for gp in graph_plots:
        label = gp.get("label")
        x = gp.get("x")   # frequency array
        y = gp.get("y")   # coherence array
        if label is None or x is None or y is None:
            continue
        band_mask = (x >= freq_min) & (x <= freq_max)
        if band_mask.sum() == 0:
            warnings.warn(f"No frequencies in [{freq_min}, {freq_max}] Hz for {fpath}")
            continue
        result[label] = float(np.mean(y[band_mask]))

    return result


def parse_filename(fpath):
    """
    Extract (subject, kinematic_measure, joint_label) from a coherence filename.
    Expected format: coherence_{subject}_{kinematic_measure}_{joint_label}_conds.json
    """
    base = op.basename(fpath).removeprefix("coherence_").removesuffix("_conds.json")
    for km in KINEMATIC_MEASURES:
        m = re.match(rf"^(.+?)_{km}_(.+)$", base)
        if m:
            return m.group(1), km, m.group(2)
    return None, None, None


# ── DataFrame builder ──────────────────────────────────────────────────────────

def build_dataframe(path_results, kinematic_measures=None, freq_min=0, freq_max=4):
    """
    Glob all *_conds.json files and build a long-format DataFrame:
    subject | kinematic_measure | joint_pair | condition | coherence
    """
    if kinematic_measures is None:
        kinematic_measures = KINEMATIC_MEASURES

    pattern = op.join(path_results, "Coherence", "Joint", "*_conds.json")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No *_conds.json files found in {op.dirname(pattern)}")

    rows = []
    for fpath in files:
        subject, km, joint_label = parse_filename(fpath)
        if subject is None or km not in kinematic_measures:
            continue
        try:
            cond_coherence = load_coherence_file(fpath, freq_min=freq_min, freq_max=freq_max)
        except Exception as e:
            warnings.warn(f"Skipping {op.basename(fpath)}: {e}")
            continue
        for condition, mean_coh in cond_coherence.items():
            rows.append({
                "subject": subject,
                "kinematic_measure": km,
                "joint_pair": joint_label,
                "condition": condition,
                "coherence": mean_coh,
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(files)} files → {len(df)} rows, "
          f"{df['subject'].nunique()} subjects, "
          f"{df['joint_pair'].nunique()} joint pairs")
    return df


# ── Statistics ─────────────────────────────────────────────────────────────────

def run_stats(df, ref_condition="Masked", alpha=0.05):
    """
    For each (kinematic_measure, joint_pair):
      1. Wilcoxon signed-rank test (paired)
      2. LMM: coherence ~ condition + (1 | subject)

    FDR correction (Benjamini-Hochberg) is applied per kinematic measure.

    Parameters
    ----------
    df : DataFrame from build_dataframe()
    ref_condition : str
        The baseline condition (default: "Masked"). The LMM coefficient and
        mean difference are reported as other_condition − ref_condition.
    alpha : float
        Significance threshold (for flagging significant pairs).

    Returns
    -------
    DataFrame sorted by kinematic_measure, then p_wilcoxon_fdr.
    """
    conditions = sorted(df["condition"].unique())
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, found: {conditions}")
    other_condition = [c for c in conditions if c != ref_condition][0]
    print(f"Conditions: reference='{ref_condition}', contrast='{other_condition}'")

    results = []
    groups = df.groupby(["kinematic_measure", "joint_pair"])

    for (km, jp), grp in groups:
        pivot = grp.pivot_table(index="subject", columns="condition", values="coherence")
        paired = pivot[[ref_condition, other_condition]].dropna()

        if len(paired) < 3:
            warnings.warn(f"Only {len(paired)} complete subjects for {km} / {jp} — skipping.")
            continue

        ref_vals   = paired[ref_condition].values
        other_vals = paired[other_condition].values

        # ── Wilcoxon signed-rank test ──────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat_w, p_w = stats.wilcoxon(ref_vals, other_vals)
            except ValueError as e:
                warnings.warn(f"Wilcoxon failed for {km}/{jp}: {e}")
                stat_w, p_w = np.nan, np.nan

        # ── LMM: coherence ~ condition + (1 | subject) ─────────────────────────
        # Explicitly set reference level so the coefficient is other − ref.
        grp = grp.copy()
        grp["condition"] = pd.Categorical(grp["condition"], categories=[ref_condition, other_condition])
        try:
            model = smf.mixedlm("coherence ~ condition", grp, groups=grp["subject"])
            fit   = model.fit(reml=True, method="lbfgs", disp=False)
            coef_key = f"condition[T.{other_condition}]"
            lmm_coef = fit.params.get(coef_key, np.nan)
            lmm_se   = fit.bse.get(coef_key, np.nan)
            p_lmm    = fit.pvalues.get(coef_key, np.nan)
        except Exception as e:
            warnings.warn(f"LMM failed for {km}/{jp}: {e}")
            lmm_coef, lmm_se, p_lmm = np.nan, np.nan, np.nan

        results.append({
            "kinematic_measure": km,
            "joint_pair":        jp,
            f"mean_{ref_condition}":   ref_vals.mean(),
            f"mean_{other_condition}": other_vals.mean(),
            "mean_diff":              other_vals.mean() - ref_vals.mean(),
            "n_subjects":             len(paired),
            "W":                      stat_w,
            "p_wilcoxon":             p_w,
            "lmm_coef":               lmm_coef,
            "lmm_se":                 lmm_se,
            "p_lmm":                  p_lmm,
        })

    results_df = pd.DataFrame(results)

    # ── FDR correction per kinematic measure ───────────────────────────────────
    for km in results_df["kinematic_measure"].unique():
        mask = results_df["kinematic_measure"] == km
        idx  = results_df[mask].index

        for p_col, fdr_col in [("p_wilcoxon", "p_wilcoxon_fdr"), ("p_lmm", "p_lmm_fdr")]:
            p_vals = results_df.loc[idx, p_col].values
            valid  = ~np.isnan(p_vals)
            if valid.sum() > 1:
                _, p_fdr, _, _ = multipletests(p_vals[valid], method="fdr_bh")
                results_df.loc[idx[valid], fdr_col] = p_fdr
            elif valid.sum() == 1:
                results_df.loc[idx[valid], fdr_col] = p_vals[valid]   # single test, no correction

    results_df["sig_wilcoxon"] = results_df["p_wilcoxon_fdr"] < alpha
    results_df["sig_lmm"]      = results_df["p_lmm_fdr"] < alpha

    return results_df.sort_values(["kinematic_measure", "p_wilcoxon_fdr"]).reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pass
    excluded_pairs = {(subject, trial) for subject, trials in EXCLUDED_RECORDINGS.items() for trial in trials}

    generate_figures = True
    filtered = True
    if USE_REREF:
        if filtered:
            df = pd.read_parquet(PATH_DATAFRAME_VELOCITY_REREF_F)
        else:
            df = pd.read_parquet(PATH_DATAFRAME_VELOCITY_REREF_U)
    else:
        if filtered:
            df = pd.read_parquet(PATH_DATAFRAME_VELOCITY_F)
        else:
            df = pd.read_parquet(PATH_DATAFRAME_VELOCITY_U)
    print(df.columns.tolist())
    df_mocap = df[(df["modality"] == "mocap") & (df["measure"] == "velocity")].copy()

    print(f"Total rows before cleaning: {len(df_mocap)}\n")
    mask = pd.MultiIndex.from_arrays([df_mocap["subject"], df_mocap["trial"]]).isin(excluded_pairs)
    df_mocap = df_mocap[~mask]
    print(f"Total rows after cleaning: {len(df_mocap)}\n")

    joints = list(df_mocap["label"].unique())  # preserve dataframe order (top-to-bottom, right-to-left)
    subjects = sorted(df_mocap["subject"].unique())
    conditions = sorted(df_mocap["condition"].unique())

    n_subjects = len(subjects)

    print(f"Joints    : {joints}")
    print(f"Subjects  : {subjects}")
    print(f"Conditions: {conditions}")

    # mean_speed_by_joint(1, df_mocap, generate_figures)
    # mean_speed_comparison_between_joints(2, df_mocap, generate_figures)
    # mean_speed_by_subject(3, df_mocap, generate_figures)
    # mean_speed_by_subject_joint(4, df_mocap, "HandRight", generate_figures)
    # mean_speed_by_subject_joint(5, df_mocap, "HandLeft", generate_figures)
    # mean_speed_by_joint_by_subject(6, df_mocap, generate_figures)
    # mean_speed_by_joint_by_subject_normalized(7, df_mocap, generate_figures)
    # mean_speed_deviation_by_joint_by_subject(8, df_mocap, generate_figures)
    # joint_rank(9, df_mocap, generate_figures)
    # joint_rank_stability(10, df_mocap, generate_figures)
    # joint_rank_by_subject(11, df_mocap, generate_figures)
    # trial_order(12, df_mocap, None, generate_figures)
    # trial_order_by_subject(13, df_mocap, None, generate_figures)
    # trial_order(14, df_mocap, "HandRight", generate_figures)
    # trial_order_by_subject(15, df_mocap, "HandRight", generate_figures)
    # trial_order(16, df_mocap, "HandLeft", generate_figures)
    # mean_speed_by_cartoon(17, df_mocap, None, generate_figures)
    # mean_speed_by_show(18, df_mocap, None, generate_figures)
    # mean_speed_by_cartoon(19, df_mocap, "HandRight", generate_figures)
    # mean_speed_by_show(20, df_mocap, "HandRight", generate_figures)
    # mean_speed_by_cartoon(21, df_mocap, "HandLeft", generate_figures)
    # mean_speed_by_condition(22, df_mocap, None, generate_figures)
    # mean_speed_by_condition(23, df_mocap, "HandRight", generate_figures)
    # mean_speed_by_condition(24, df_mocap, "HandLeft", generate_figures)
    # mean_speed_by_condition(25, df_mocap, "Head", generate_figures)
    # mean_speed_by_condition_by_subject(26, df_mocap, None, generate_figures)
    # mean_speed_by_condition_by_subject(27, df_mocap, "HandRight", generate_figures)
    # mean_speed_by_condition_by_subject(28, df_mocap, "Head", generate_figures)
    # mean_speed_difference_by_condition(29, df_mocap, generate_figures)
    # bilateral_symmetry(30, df_mocap, generate_figures)
    # bilateral_symmetry_by_condition(31, df_mocap, generate_figures)
    # joint_correlation_matrix(32, df_mocap, generate_figures)
    # joint_correlation(33, df_mocap, "HandRight", generate_figures)
    # joint_correlation(34, df_mocap, "Head", generate_figures)
    # joint_coherence_body_graph(35, "HandRight", "velocity", "raw", "subject",
    #                            generate_figure=generate_figures)
    joint_coherence_silhouette(36, "HandRight", "velocity", [0.2], "raw", generate_figures)
    # joint_coherence_silhouette(37, "HandRight", "velocity", [0.5, 1], "average", generate_figures)
    # joint_coherence_body_graph(38, "HandLeft", "velocity", "raw", "subject", generate_figure=generate_figures)
    joint_coherence_silhouette(39, "HandLeft", "velocity", [0.2], "raw", generate_figures)
    # joint_coherence_silhouette(40, "HandLeft", "velocity", [0.5, 1], "average", generate_figures)
    # joint_coherence_body_graph(41, "Head", "velocity", "raw", "subject", generate_figure=generate_figures)
    joint_coherence_silhouette(42, "Head", "velocity", [0.2], "raw", generate_figures)

    # generate_figures = True
    # i = 35
    # frequency = [0.2]
    #
    # joints = ["HandRight", "HandLeft", "Head"]
    # value_types = ["raw", "z-score"]
    # levels = ["subject", "group"]
    #
    # for joint, value_type, level in product(joints, value_types, levels):
    #     if value_type == "raw":
    #         joint_coherence_body_graph(i, joint, "velocity", value_type, level, None, generate_figures)
    #         joint_coherence_silhouette(i + 1, joint, "velocity", frequency, value_type, level, generate_figures)
    #     i += 2

    # test_coherence_at_frequency("HandRight", "HandLeft", "velocity", 1)
    # joint_coherence_ttest_body_graph(43, "HandRight", "velocity")
    # freq_bands = {f"{i}–{i + 1} Hz": (i, i + 1) for i in range(4)}
    # df_long = joint_coherence_lmm(44, "HandRight", "velocity", 4)
    # joint_coherence_lmm_plots(45 df_long, "HandRight", "velocity")
    # joint_coherence_matrix(46, "velocity", 0.5, generate_figures)
    # joint_coherence_matrix(47, "velocity", 1, generate_figures)
    # power_spectrum_analysis(48, df_mocap, "velocity", generate_figures,
    #                         average="subjects",
    #                         series=None)
    # power_spectrum_analysis(49, df_mocap, "velocity", generate_figures,
    #                         average="subjects",
    #                         series=None,
    #                         result_type="z-scores",
    #                         permutation_method="shift",
    #                         number_of_randperms=1000,
    #                         fit_background=False,
    #                         signif_style="threshold")

    # TEST PSD
    # df = _make_synthetic_psd_df()
    # plot_psd(50, "velocity", "envelope", df, use_reref=True,
    #          subjects = None, trials = None, xlim = [0, 4], legend = True, show_average = False,
    #          joints = "HandRight", joint_layout = "HandRight",
    #          nperseg = 256, normalization = "fooof", show = True)
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False,
    #                  "font.family": "Metropolis", "xtick.labelsize": 8, "ytick.labelsize": 8}
    # sns_kwargs = {"style": "whitegrid", "font_scale": 1.2, "rc": custom_params}
    # mpl_kwargs = {"hspace": 1, "left": 0.07, "right": 0.93, "top": 0.93, "bottom": 0.07}
    # plot_psd(51, "velocity", "envelope", use_reref=True,
    #          subjects = None, trials = None, xlim = [0, 4], legend = False, show_average = None,
    #          joints = UPPER_BODY_JOINTS, joint_layout = [[None, None, "Head", None, None],
    #                                         [None, None, "Neck", None, None],
    #                                         ["ElbowRight", "ShoulderRight", "SpineShoulder", "ShoulderLeft", "ElbowLeft"],
    #                                         ["WristRight", None, "SpineMid", None, "WristLeft"],
    #                                         ["HandRight", "HipRight", "SpineBase", "HipLeft", "HandLeft"]],
    #          x_label = "Frequency (Hz)", y_label = "Normalized amplitude",
    #          nperseg = 256, normalization = "subject_peak", sns_kwargs = sns_kwargs,
    #          mpl_kwargs = mpl_kwargs, show = True, freq_log=False,
    #          path_png_silhouette=op.join(PATH_FIGURES, "fig37_joint_power.png"))
    # plot_psd(52, "velocity", "envelope",
    #          joints = ["Head", "HandRight", "HandLeft", "HipRight", "HipLeft", "FootRight", "FootLeft"],
    #          fs=20, freq_max=4, normalize="total_power", generate_figure=generate_figures)
    # pca_analysis(53, df_mocap, "velocity", n_components=3, generate_figure=generate_figures)

    # custom_params = {"axes.spines.right": False, "axes.spines.top": False,
    #                  "font.family": "Metropolis", "xtick.labelsize": 8, "ytick.labelsize": 8}
    # sns_kwargs = {"style": "whitegrid", "font_scale": 1.2, "rc": custom_params}
    # mpl_kwargs = {"hspace": 1, "left": 0.07, "right": 0.93, "top": 0.93, "bottom": 0.07}
    # plot_psd(54, "velocity", "envelope", use_reref=True,
    #          subjects = None, trials = None, xlim = [0, 4], legend = False, show_average = None,
    #          joints = UPPER_BODY_JOINTS, joint_layout = [[None, None, "Head", None, None],
    #                                         [None, None, "Neck", None, None],
    #                                         ["ElbowRight", "ShoulderRight", "SpineShoulder", "ShoulderLeft", "ElbowLeft"],
    #                                         ["WristRight", None, "SpineMid", None, "WristLeft"],
    #                                         ["HandRight", "HipRight", "SpineBase", "HipLeft", "HandLeft"]],
    #          x_label = "Frequency (Hz)", y_label = "Normalized amplitude", file_trail="_hp0p2Hz",
    #          nperseg = 256, normalization = "subject_peak", sns_kwargs = sns_kwargs,
    #          mpl_kwargs = mpl_kwargs, show = True, freq_log=False,
    #          path_png_silhouette=None)
    # plot_psd(55, "velocity", "envelope",
    #          joints = ["Head", "HandRight", "HandLeft", "HipRight", "HipLeft", "FootRight", "FootLeft"],
    #          fs=20, freq_max=4, normalize="total_power", generate_figure=generate_figures)
    # pca_analysis(56, df_mocap, "velocity", n_components=3, generate_figure=generate_figures)

    # df = build_dataframe(PATH_RESULTS)
    #
    # # Optionally restrict to specific measures here, e.g.:
    # # df = df[df["kinematic_measure"] == "velocity"]
    #
    # results = run_stats(df)
    #
    # # ── Print summary ──────────────────────────────────────────────────────────
    # for km, grp in results.groupby("kinematic_measure"):
    #     n_sig_w = grp["sig_wilcoxon"].sum()
    #     n_sig_lmm = grp["sig_lmm"].sum()
    #     print(f"\n── {km} ({len(grp)} joint pairs) "
    #           f"── Wilcoxon sig (FDR): {n_sig_w}  |  LMM sig (FDR): {n_sig_lmm}")
    #     sig = grp[grp["sig_wilcoxon"] | grp["sig_lmm"]]
    #     if not sig.empty:
    #         print(sig[["joint_pair", "mean_diff", "W", "p_wilcoxon_fdr",
    #                    "lmm_coef", "p_lmm_fdr"]].to_string(index=False))
    #
    # # ── Save per kinematic measure ─────────────────────────────────────────────
    # out_dir = op.join(PATH_RESULTS, "Coherence", "Joint")
    # for km, grp in results.groupby("kinematic_measure"):
    #     out_path = op.join(out_dir, f"joint_coherence_stats_{km}.csv")
    #     grp.to_csv(out_path, index=False, float_format="%.6f")
    #     print(f"Saved → {out_path}")