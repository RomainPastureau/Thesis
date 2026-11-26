import math
import os
import pandas as pd
import numpy as np
from krajjat import Graph, GraphPlot
from scipy import stats
from scipy.signal import welch
from colorama import Fore, Style
import pingouin as pg
from krajjat.analysis_functions import pca
from krajjat.plot_functions import plot_silhouette, plot_body_graphs
from scipy.stats import spearmanr

from constants import *
from helper_functions import save_figure, print_section_title
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.formula.api as smf
from itertools import combinations, product
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
import pickle
import hashlib

PATH_DATAFRAME_VELOCITY_F = op.join(PATH_DATAFRAME, f"FS{int(FS)} filtered", f"{SYSTEM.lower()}_velocity_envelope.gzip")
PATH_DATAFRAME_VELOCITY_REREF_F = op.join(PATH_DATAFRAME, f"FS{int(FS)} filtered", f"{SYSTEM.lower()}_velocity_envelope{REREF_SUFFIX}.gzip")
PATH_DATAFRAME_VELOCITY_U = op.join(PATH_DATAFRAME, f"FS{int(FS)} unfiltered", f"{SYSTEM.lower()}_velocity_envelope.gzip")
PATH_DATAFRAME_VELOCITY_REREF_U = op.join(PATH_DATAFRAME, f"FS{int(FS)} unfiltered", f"{SYSTEM.lower()}_velocity_envelope{REREF_SUFFIX}.gzip")

PATH_DATAFRAME_ACCELERATION_F = op.join(PATH_DATAFRAME, f"FS{int(FS)} filtered", f"{SYSTEM.lower()}_acceleration_envelope.gzip")
PATH_DATAFRAME_ACCELERATION_REREF_F = op.join(PATH_DATAFRAME, f"FS{int(FS)} filtered", f"{SYSTEM.lower()}_acceleration_envelope{REREF_SUFFIX}.gzip")
PATH_DATAFRAME_ACCELERATION_U = op.join(PATH_DATAFRAME, f"FS{int(FS)} unfiltered", f"{SYSTEM.lower()}_acceleration_envelope.gzip")
PATH_DATAFRAME_ACCELERATION_REREF_U = op.join(PATH_DATAFRAME, f"FS{int(FS)} unfiltered", f"{SYSTEM.lower()}_acceleration_envelope{REREF_SUFFIX}.gzip")


PATH_FIGURES = op.join(PATH_PLOTS, "Kinematics")
os.makedirs(PATH_FIGURES, exist_ok=True)

USE_REREF = True

# ── Coherence result paths (Group / Per subject, gzip layout) ────────────────
PATH_COHERENCE_GROUP = op.join(PATH_RESULTS, "Coherence", "Joint", "Group")
PATH_COHERENCE_SUBJECT = op.join(PATH_RESULTS, "Coherence", "Joint", "Per subject")


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
                        font_size=0.5,
                        verbosity=0)


def mean_speed_comparison_between_joints(n, df, generate_figure=True):
    print_section_title(f"{n}. COMPARISON OF THE MEAN SPEED BETWEEN JOINTS")

    joints = list(df_mocap["label"].unique())
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
                            fontsize=4.5, color="white")

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

    # Overall mean speed per subject (collapsed across joints and trials)
    overall_mean = df_joint.groupby("subject")["value"].mean()
    print("\nOverall mean speed per subject (across all joints):")
    overall_mean = overall_mean * 100
    print(overall_mean.round(2).to_string())
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
        n_cols_grid = 6
        plot_silhouette(plot_dict_subj,
                        title="Mean speed per joint per subject",
                        title_silhouette=[s for s in subjects],
                        title_scale="Mean speed (m/s)",
                        min_scale=0, max_scale=float(mean_speed_subj.max().max()),
                        n_cols=n_cols_grid,
                        show=False,
                        color_scheme="celsius",
                        path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_mean_speed_by_joint_by_subject.png"),
                        resolution=(1200, 1080),
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
                        title_silhouette=[s for s in subjects],
                        title_scale="Ratio to subject mean",
                        min_scale=0, max_scale=vmax,
                        n_cols=6,
                        color_scheme="celsius",
                        show=False,
                        path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_normalized_speed_by_subject.png"),
                        resolution=(1200, 1080),
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
    icc_result = pg.intraclass_corr(data=icc_long, targets="joint", raters="subject", ratings="ratio",
                                    nan_policy="omit")
    print("\nICC (consistency of normalized recruitment pattern across subjects):")
    print(icc_result[["Type", "ICC", "CI95", "pval"]].to_string())

    if generate_figure:
        vmax = float(deviation.replace([np.inf, -np.inf], np.nan).abs().max().max())

        plot_dict_dev = {joint: deviation[joint].values
                         for joint in joints if joint in deviation.columns}

        plot_silhouette(plot_dict_dev,
                        title="Deviation from group mean normalized speed",
                        title_silhouette=[s for s in subjects],
                        title_scale="Deviation from group mean",
                        min_scale=-vmax, max_scale=vmax,
                        color_scheme=["#2166ac", "#f7f7f7", "#d6604d"],
                        n_cols=6,
                        show=False,
                        path_save=op.join(PATH_FIGURES,
                                          f"fig{str(n).zfill(2)}_joint_mean_speed_deviation_from_group_mean.png"),
                        resolution=(1200, 1080),
                        verbosity=0)


def joint_rank(n, df, generate_figure=True):
    print_section_title(f"{n}. OVERALL JOINT RANK")

    mean_speed = df.groupby("label")["value"].mean().reindex(joints)
    mean_speed_subj = (df.groupby(["subject", "label"])["value"]
                       .mean().unstack("label").reindex(columns=joints))
    rank_per_subject = (mean_speed_subj.rank(axis=1, ascending=False, na_option="bottom").round(0).astype("Int64"))
    mean_rank = rank_per_subject.mean().reindex(joints)

    print("\nOverall joint ranking by mean speed:")
    ranking = mean_speed.sort_values(ascending=False)
    for i, (joint, val) in enumerate(ranking.items()):
        print(f"  {i + 1:2d}. {joint:<20} {val:.4f}")

    print("\nMean rank per joint across subjects (lower = more dominant):")
    for joint, val in mean_rank.sort_values().items():
        print(f"  {joint:<20} {val:.2f}")

    if USE_REREF:
        rank_matrix = rank_per_subject[[j for j in joints if j != "SpineBase"]].values
    else:
        rank_matrix = rank_per_subject.values
    W, chi2, p = kendalls_w(rank_matrix)
    interp = "excellent" if W > 0.9 else "good" if W > 0.7 else "moderate" if W > 0.5 else "poor"
    sig = Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p < 0.05 else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
    print(f"\nKendall's W (consistency of joint ranking across subjects):")
    print(f"  W = {W:.3f}, chi2({len(joints) - 2}) = {chi2:.3f}, p = {p:.4f} — {interp} {sig}")

    if generate_figure:
        n_joints = len(joints)
        overall_rank = mean_speed.rank(ascending=False).astype(int)
        plot_silhouette({joint: int(overall_rank[joint]) for joint in joints},
                        title="Joint rank across subjects",
                        title_scale="Overall rank",
                        min_scale=1, max_scale=float(n_joints),
                        color_scheme="celsius_r",
                        font_size=0.5,
                        show=False,
                        path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_overall_joint_rank.png"),
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
    rank_per_subject = (mean_speed_subj.rank(axis=1, ascending=False, na_option="bottom").round(0).astype("Int64"))

    n_missing = mean_speed_subj.isna().sum().sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing} missing joint/subject combinations — ranked to bottom")

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
    data = rank_per_subject[joints_for_test].values.astype(float)
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
    rank_per_subject = (mean_speed_subj.rank(axis=1, ascending=False, na_option="bottom").round(0).astype("Int64"))
    print(rank_per_subject.to_string())

    n_missing = mean_speed_subj.isna().sum().sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing} missing joint/subject combinations — ranked to bottom")

    overall_rank = mean_speed.rank(ascending=False)
    print("\nSpearman correlation of each subject's rank vs. overall rank:")
    for subj in subjects:
        subj_rank = rank_per_subject.loc[subj]
        r, p = spearmanr(subj_rank, overall_rank)
        print(f"  {subj}: r = {r:.3f}, p = {p:.4f}")

    if generate_figure:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(rank_per_subject.astype(float), ax=ax, cmap="YlOrRd_r", annot=True, fmt=".0f",
                    linewidths=0.5, cbar_kws={"label": "Rank (1 = most dominant)"}, annot_kws={"fontsize":8})
        ax.set_title("Joint rank per subject (1 = fastest)")
        ax.set_xlabel("Joint")
        ax.set_ylabel("Subject")
        plt.xticks(rotation=90, fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
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
            line_color = "#808080"
            text_color = "#808080"
            fontweight = "normal"
        else:
            line_color = "black"
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
        #         color=line_color, alpha=0.3, linewidth=1.5, linestyle="--",
        #         label=f"LMM: β = {beta:.5f}, p = {lmm_p:.3f}")
        ax.plot(x_lmm, (m_all * x_lmm + b_all) * 100, color="black", alpha=0.3, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Trial order")
        ax.set_ylabel("Mean speed (cm/s)")
        # ax.set_title(f"Group-level trial order trend — {joint_label} (mean ± SEM across subjects)")
        y_range = group_trend["mean"].max() - group_trend["mean"].min()
        y_ref = (m_all * x_lmm[-1] + b_all) * 100
        ax.text(x_lmm[-1], y_ref - y_range * 0.00, f"r = {fmt_nolead(r_all)}",
                ha="right", va="bottom", fontsize=12, fontweight=fontweight, color=text_color)
        ax.text(x_lmm[-1], y_ref - y_range * 0.13, f"p {fmt_p(p_all)}",
                ha="right", va="top", fontsize=12, fontweight=fontweight, color=text_color)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if not external_ax:
            fig.tight_layout()
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_trial_order_group_{joint_label}.png", **SAVE_KW)
            plt.close(fig)

    return r_all, p_all, lmm_p

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
        n_cols = 6
        n_rows = math.ceil(n_subj / n_cols)

        fig, axes = plt.subplots(4, 6, figsize=(20, 14), sharey=True)
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
            ax.set_title(f"{subj}  (r = {r_s:.2f}, {sig})", fontsize=11)
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

    # Load show mapping from the info Excel file
    df_info = pd.read_excel(PATH_INFO)
    cartoon_to_show = df_info.drop_duplicates("Retelling").set_index("Retelling")["Show"]

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

    print(f"Descriptive statistics per cartoon (N = {len(by_cartoon)} cartoons) [{joint_label}]:")
    print(by_cartoon.to_string())

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

    # grand_mean = trial_means.groupby("cartoon")["mean_speed"].mean().mean()
    # cartoon_means = trial_means.groupby("cartoon")["mean_speed"].mean()
    # ref_cartoon = (cartoon_means - grand_mean).abs().idxmin()
    # print(f"\nReference cartoon (closest to grand mean): {ref_cartoon} "
    #       f"({cartoon_means[ref_cartoon]:.4f} m/s, grand mean = {grand_mean:.4f} m/s)")
    #
    # # LMM: mean_speed ~ cartoon + (1|subject)
    # # Cartoon is categorical so use C() notation
    # print("\nLinear Mixed Model: mean_speed ~ cartoon + (1|subject)")
    # lmm = smf.mixedlm(f"mean_speed ~ C(cartoon, Treatment('{ref_cartoon}'))",
    #                    trial_means, groups=trial_means["subject"])
    # lmm_fit = lmm.fit(reml=True)
    # print(lmm_fit.summary())
    #
    # # Extract p-values for each cartoon level
    # print("\nCartoon fixed effects:")
    # for param, beta, p in zip(lmm_fit.params.index,
    #                           lmm_fit.params.values,
    #                           lmm_fit.pvalues.values):
    #     if "cartoon" in param.lower():
    #         sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p < 0.05
    #                else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    #         print(f"  {param}: β = {beta:.4f}, p = {p:.4f} {sym}")

    groups = [grp["mean_speed"].values for _, grp in trial_means.groupby("cartoon")]
    k = len(groups)
    n_total = len(trial_means)

    h_stat, p_kruskal = stats.kruskal(*groups)
    eta2 = h_stat / (n_total - 1)

    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_kruskal < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"\nKruskal-Wallis: H({k - 1}) = {h_stat:.3f}, p = {p_kruskal:.4f}  {sym}")
    print(f"Effect size: η² = {eta2:.3f}")

    print("\nLMM likelihood ratio test (controlling for subject):")
    lmm_null = smf.mixedlm("mean_speed ~ 1", trial_means,
                           groups=trial_means["subject"]).fit(reml=False)
    lmm_full = smf.mixedlm("mean_speed ~ C(cartoon)", trial_means,
                           groups=trial_means["subject"]).fit(reml=False)
    lr_stat = 2 * (lmm_full.llf - lmm_null.llf)
    from scipy.stats import chi2
    p_lr = chi2.sf(lr_stat, df=k - 1)
    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_lr < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"  χ²({k - 1}) = {lr_stat:.3f}, p = {p_lr:.4f} {sym}")

    if p_kruskal < 0.05:
        import scikit_posthocs as sp
        dunn = sp.posthoc_dunn(trial_means, val_col="mean_speed",
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
                          markersize=6, label=s)
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
    df_info = pd.read_excel(PATH_INFO)
    cartoon_to_show = df_info.drop_duplicates("Retelling").set_index("Retelling")["Show"]

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
    TOP_PAIRS = ["HandIn", "HandOut", "WristIn", "WristOut", "Elbow", "Arm", "ShoulderTop", "ShoulderBack", "Knee"]
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
            mean_ai = ((r_vals - l_vals) / (r_vals + l_vals)).mean()

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
        subj_data = subj_data.fillna(subj_data.mean())
        remaining_nan = subj_data["HandInRight"].isna().sum()
        if remaining_nan > 0:
            print(f"  ⚠ {subj}: HandInRight still has {remaining_nan} NaN after fillna!")

        r_mat, _ = spearmanr(subj_data.values)
        if r_mat.ndim == 0:
            continue
        r_mat = np.array(r_mat)
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
            subj_data = trial_speed.xs(subj, level="subject").dropna(how="all")
            subj_data = subj_data.fillna(subj_data.mean()).values
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
        subj_data = trial_speed.xs(subj, level="subject").dropna(how="all")
        subj_data = subj_data.fillna(subj_data.mean())
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

        r_dict = {j: (v if np.isfinite(v) else 0.0) for j, v in r_dict.items()}

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
            font_size=0.5,
            show=False,
            path_save=op.join(PATH_FIGURES,
                              f"fig{str(n).zfill(2)}_joint_correlation_{target_joint}.png"),
            resolution=(800, 1080),
            verbosity=0,
        )

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
        subjects = SUBJECTS_NO_OUTLIERS
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


def joint_coherence_body_graph(n, target_joint, sequence_measure, value_type="z-score", level="subject",
                               gesture_band=None, generate_figure=True, subjects_to_consider=ALL_SUBJECTS):
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
                for subject in subjects_to_consider:
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

        def _annotate_reref_axis(fig, joint_name="SpineBase", hatch="////", text="Re-referencing joint"):
            import matplotlib.patches as mpatches

            for ax in fig.axes:
                if ax.get_title() == joint_name:
                    rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, hatch=hatch, facecolor="white",
                        edgecolor="lightgray", linewidth=0.5, zorder=0, )
                    ax.add_patch(rect)
                    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontstyle="italic",
                        fontsize=7, color="gray", zorder=2, )
                    break

        mpl_kwargs = {"hspace": 1, "tick_pad": -4, "xlabel_pad": -1, "ylabel_pad": 0, "title_pad": 3,
                      "title_weight": "bold", "xtick_step": 1, "tight_layout": False}

        fig = plot_body_graphs(plot_dict, title=None, sns_kwargs={"style": "whitegrid",
                                                                  "rc": {"font.family": "Metropolis",
                                                                         "axes.spines.right": False,
                                                                         "axes.spines.top": False}},
            mpl_kwargs=mpl_kwargs, xlim=[0, 4], joint_layout="Qualisys", legend=False, show=False, path_save=None,
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
                               generate_figure=True, subjects_to_consider=ALL_SUBJECTS):
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

    data_by_joint = {joint: arr_by_lag for joint, arr_by_lag in data_by_joint.items() if joint in ALL_JOINTS}

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
        for j in UPPER_BODY_JOINTS:
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
                color_scheme="celsius", show_values=True, value_format=".2f", strip_leading_zero=True, font_size=0.5,
                show=False, path_save=op.join(PATH_FIGURES, f"fig{str(n).zfill(2)}_joint_coherence_{target_joint}"
                                                            f"_{freq_suffix}{trailing_file_name}_{value_type}_{level}.png"),
                path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"), resolution=(600, 1080), verbosity=0, )
            print(f"  Saved silhouette: {freq_suffix}")

        else:
            val_grid = {j: np.array([_build_val_dict(indices)[j] for _, indices in freq_specs]) for j in
                        UPPER_BODY_JOINTS}

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


# def joint_coherence_silhouette(n, target_joint, sequence_measure, subjects, frequency=None,
#                                kind="average", generate_figure=True):
#     from krajjat import Results
#     import glob
#
#     if USE_REREF:
#         trailing_file_name = "_rWaistBack"
#     else:
#         trailing_file_name = ""
#
#     print_section_title(f"{n}. JOINT COHERENCE — {target_joint.upper()}")
#
#     coherence_dir = op.join(PATH_RESULTS, "Coherence", "Joint")
#     pattern = op.join(coherence_dir,
#                       f"coherence_*_{sequence_measure}_{target_joint}{trailing_file_name}.json")
#     subject_files = sorted(glob.glob(pattern))
#
#     if not subject_files:
#         print(f"\n  No subject files found matching pattern:\n  {pattern}")
#         return
#
#     prefix = f"coherence_"
#     suffix = f"_{sequence_measure}_{target_joint}{trailing_file_name}.json"
#     subject_names = [op.basename(f)[len(prefix):-len(suffix)] for f in subject_files]
#     print(f"\n  Found {len(subject_names)} subject(s): {subject_names}")
#
#     # ── load all subjects and average per joint ────────────────────────────────
#     stacked = {}
#     out_ref = None
#
#     for subject, path_json in zip(subjects, subject_files):
#         print(f"  Loading [{subject}]: {path_json}")
#         out = Results.load(path_json)
#
#         if out_ref is None:
#             out_ref = out
#
#         key    = (target_joint, sequence_measure)
#         sm_key = out.sequence_measures[0]
#         sv     = out.series_values[0]
#         lag    = out.lags[0]
#
#         if kind == "z-score":
#             data_by_joint = out.z_scores[key][sm_key][sv]
#         else:
#             data_by_joint = out.averages[key][sm_key][sv]
#
#         for joint, arr_by_lag in data_by_joint.items():
#             stacked.setdefault(joint, []).append(arr_by_lag[lag])
#
#     data_by_joint = {joint: {lag: np.nanmean(arrays, axis=0)}
#                      for joint, arrays in stacked.items()
#                      if len(arrays) > 0 and joint in joints}
#
#     key    = (target_joint, sequence_measure)
#     sm_key = out_ref.sequence_measures[0]
#     sv     = out_ref.series_values[0]
#     lag    = out_ref.lags[0]
#     freqs  = np.array(out_ref.frequencies)
#
#     value_label = "z-score"   if kind == "z-score" else "coherence"
#     scale_title = "Z-score"   if kind == "z-score" else "Coherence"
#     valid_joints = [j for j in data_by_joint if not np.all(np.isnan(data_by_joint[j][lag]))]
#
#     # ── build freq_specs: list of (display_label, indices_array | None) ────────
#     # indices_array = None  → mean across ALL frequencies
#     # indices_array = array → mean across those specific bins (single or range)
#
#     def _closest_idx(f):
#         return int(np.argmin(np.abs(freqs - f)))
#
#     def _range_indices(lo, hi):
#         return np.where((freqs >= lo) & (freqs <= hi))[0]
#
#     def _range_label(indices):
#         return f"{freqs[indices[0]]:.2f}–{freqs[indices[-1]]:.2f} Hz"
#
#     def _label_to_suffix(label):
#         return (label.replace(" Hz", "hz").replace("–", "_to_")
#                      .replace(".", "_").replace(" ", "_"))
#
#     def _parse_freq_entry(f):
#         if isinstance(f, tuple):
#             lo, hi = f
#             idx = _range_indices(lo, hi)
#             return _range_label(idx), idx
#         else:
#             idx = np.array([_closest_idx(f)])
#             return f"{freqs[idx[0]]:.3f} Hz", idx
#
#     if frequency is None:
#         freq_specs = [("mean", None)]
#     elif isinstance(frequency, tuple):
#         freq_specs = [_parse_freq_entry(frequency)]
#     elif isinstance(frequency, (int, float)):
#         freq_specs = [_parse_freq_entry(frequency)]
#     else:
#         freq_specs = [_parse_freq_entry(f) for f in frequency]
#
#     # ── val_dict builder ───────────────────────────────────────────────────────
#     def _build_val_dict(indices):
#         val_dict = {}
#         for joint in data_by_joint:
#             arr = data_by_joint[joint][lag]
#             if np.all(np.isnan(arr)):
#                 val_dict[joint] = 0.0
#             elif indices is None:
#                 val_dict[joint] = float(np.nanmean(arr))
#             else:
#                 val_dict[joint] = float(np.nanmean(arr[indices]))
#         for j in joints:
#             val_dict.setdefault(j, 0.0)
#         return val_dict
#
#     # ── print summary ──────────────────────────────────────────────────────────
#     print(f"\n{value_label.capitalize()} — coherence with {target_joint}:")
#     for label, indices in freq_specs:
#         val_dict = _build_val_dict(indices)
#         print(f"\n  @ {label}:")
#         print(f"  {'Joint':<22} {value_label.capitalize():>10}")
#         print("  " + "─" * 34)
#         for joint, val in sorted(
#                 [(j, val_dict[j]) for j in valid_joints if j != target_joint],
#                 key=lambda x: -x[1]):
#             bar = "█" * min(int(val / 10), 30)
#             print(f"  {joint:<22} {val:>10.2f}  {bar}")
#         print(f"  {target_joint:<22} {'(ref)':>10}")
#
#     if generate_figure:
#         if len(freq_specs) == 1:
#             label, indices = freq_specs[0]
#             val_dict = _build_val_dict(indices)
#
#             other_vals = [v for j, v in val_dict.items() if j != target_joint]
#             val_max = max((v for v in other_vals if np.isfinite(v)), default=1.0)
#             val_max = max(val_max, 1.0)
#
#             freq_suffix  = "mean" if indices is None else _label_to_suffix(label)
#             title_freq   = (f"mean {value_label} (all frequencies)" if indices is None
#                             else f"mean {value_label} ({label})" if len(indices) > 1
#                             else f"{value_label} @ {label}")
#
#             plot_silhouette(
#                 val_dict,
#                 title=None,
#                 title_scale=scale_title,
#                 min_scale=0, max_scale=val_max,
#                 color_scheme="celsius",
#                 show_values=True, values_ndigits=2,
#                 font_size=0.5, show=False,
#                 path_save=op.join(PATH_FIGURES,
#                                   f"fig{str(n).zfill(2)}_joint_coherence_{target_joint}"
#                                   f"_{freq_suffix}{trailing_file_name}_{kind}.png"),
#                 path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"),
#                 resolution=(600, 1080),
#                 verbosity=0,
#             )
#             print(f"  Saved silhouette: {freq_suffix}")
#
#         else:
#             val_grid = {j: np.array([_build_val_dict(indices)[j]
#                                      for _, indices in freq_specs])
#                         for j in joints}
#
#             all_vals = [v for j in val_grid for v in val_grid[j]
#                         if j != target_joint and np.isfinite(v)]
#             val_max = max(all_vals, default=1.0)
#             val_max = max(val_max, 1.0)
#
#             title_silhouettes = [label for label, _ in freq_specs]
#
#             first_label = freq_specs[0][0]
#             last_label  = freq_specs[-1][0]
#             freq_suffix = (
#                 "mean" if freq_specs[0][1] is None
#                 else f"{_label_to_suffix(first_label)}_to_{_label_to_suffix(last_label)}"
#             )
#
#             n_cols = min(len(freq_specs), 4)
#
#             plot_silhouette(
#                 val_grid,
#                 title=f"Coherence with {target_joint} — {value_label}s by frequency",
#                 title_silhouette=title_silhouettes,
#                 title_scale=scale_title,
#                 min_scale=0, max_scale=val_max,
#                 n_cols=n_cols,
#                 color_scheme="celsius",
#                 show_values=True, values_ndigits=1,
#                 font_size=0.8, show=False,
#                 path_save=op.join(PATH_FIGURES,
#                                   f"fig{str(n).zfill(2)}_joint_coherence_{target_joint}"
#                                   f"_{freq_suffix}{trailing_file_name}_{kind}.png"),
#                 resolution=(800 + 200 * n_cols, 1080),
#                 verbosity=0,
#             )
#             print(f"  Saved grid silhouette: {len(freq_specs)} frequencies")


def joint_coherence_matrix(n, sequence_measure, frequency=None, generate_figure=True):
    from krajjat import Results

    if USE_REREF:
        trailing_file_name = "_rWaistBack"
    else:
        trailing_file_name = ""

    print_section_title(f"{n}. JOINT COHERENCE MATRIX")

    joints_test = [j for j in joints if j != "SpineBase"] if USE_REREF else joints.copy()
    n_joints = len(joints_test)

    # ── Resolve frequency parameter to a scalar extractor ────────────────────
    # frequency: None   → mean z across all freqs
    #            float  → z at closest single frequency
    #            tuple  → mean z within (lo, hi) band

    def _extract_scalar(z_arr, p_arr, freqs):
        """Return (z_scalar, z_threshold) for this joint's arrays."""
        if np.all(np.isnan(z_arr)):
            return np.nan, np.nan
        if frequency is None:
            mask = np.ones(len(freqs), dtype=bool)
        elif isinstance(frequency, tuple):
            lo, hi = frequency
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            idx = int(np.argmin(np.abs(freqs - frequency)))
            mask = np.zeros(len(freqs), dtype=bool)
            mask[idx] = True
        return float(np.nanmean(z_arr[mask])), float(
            np.nanmean(z_arr[~p_arr_lt_alpha(p_arr, mask)]) if np.any(~p_arr_lt_alpha(p_arr, mask)) else np.nan)

    def p_arr_lt_alpha(p_arr, mask):
        """Boolean mask of significant bins within the frequency selection."""
        return (p_arr < 0.05) & mask

    def _z_threshold_from(z_arr, p_arr, freqs):
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

    def _z_scalar(z_arr, freqs):
        """Extract scalar z for this pair given frequency parameter."""
        if np.all(np.isnan(z_arr)):
            return np.nan
        if frequency is None:
            return float(np.nanmean(z_arr))
        elif isinstance(frequency, tuple):
            lo, hi = frequency
            mask = (freqs >= lo) & (freqs <= hi)
            return float(np.nanmean(z_arr[mask]))
        else:
            idx = int(np.argmin(np.abs(freqs - frequency)))
            return float(z_arr[idx])

    # ── Load all result files and build z-matrix ──────────────────────────────
    print(f"\nLoading coherence results for {n_joints} joints...")

    z_matrix = pd.DataFrame(np.nan, index=joints_test, columns=joints_test)
    thresh_vals = []  # for deriving group threshold

    for target_joint in joints_test:
        path_json = op.join(PATH_RESULTS, "Coherence", "Joint",
                            f"coherence_{sequence_measure}_{target_joint}"
                            f"{trailing_file_name}.json")
        if not op.exists(path_json):
            print(f"  ⚠ Missing: {path_json}")
            continue

        out = Results.load(path_json)
        key = (target_joint, sequence_measure)

        # Fallback for files saved during the broken period (sequence_measures/series_values empty)
        if out.sequence_measures and out.series_values:
            sm_key = out.sequence_measures[0]
            sv = out.series_values[0]
        else:
            # Recover from the z_scores structure directly
            sm_key = next(iter(out.z_scores[key]))
            sv = next(iter(out.z_scores[key][sm_key]))
            print(f"  ℹ Recovered sm_key={sm_key!r}, sv={sv!r} from z_scores for {target_joint}")

        lag = out.lags[0]
        freqs = np.array(out.frequencies)
        z_by_joint = out.z_scores[key][sm_key][sv]
        p_by_joint = out.p_values[key][sm_key][sv]

        for other_joint in joints_test:
            if other_joint not in z_by_joint:
                continue
            z_arr = z_by_joint[other_joint][lag]
            z_matrix.loc[target_joint, other_joint] = _z_scalar(z_arr, freqs)

        # Collect boundary-z values for threshold estimation
        for other_joint in joints_test:
            if other_joint == target_joint or other_joint not in z_by_joint:
                continue
            z_arr = z_by_joint[other_joint][lag]
            p_arr = p_by_joint[other_joint][lag]
            if np.all(np.isnan(z_arr)):
                continue
            t = _z_threshold_from(z_arr, p_arr, freqs)
            if np.isfinite(t):
                thresh_vals.append(t)

        print(f"  ✓ {target_joint}")

    # Diagonal is meaningless (self-coherence) — set to NaN
    for j in joints_test:
        z_matrix.loc[j, j] = np.nan

    # ── Significance threshold ────────────────────────────────────────────────
    z_threshold = float(np.median(thresh_vals)) if thresh_vals else 1.96
    sig_matrix = z_matrix > z_threshold

    # ── Print summary ─────────────────────────────────────────────────────────
    freq_label = (
        "all frequencies" if frequency is None
        else f"{frequency[0]}–{frequency[1]} Hz" if isinstance(frequency, tuple)
        else f"{frequency} Hz"
    )
    print(f"\nFrequency: {freq_label}")
    print(f"Significance threshold z = {z_threshold:.3f}  "
          f"(median boundary, α = {out.signif_alpha}, "
          f"n_perms = {out.number_of_randperms})")
    n_sig = int(sig_matrix.values[~np.eye(n_joints, dtype=bool)].sum()) // 2
    n_total = n_joints * (n_joints - 1) // 2
    print(f"Significant pairs: {n_sig}/{n_total}")

    print(f"\n  {'Joint':<22} {'Mean z (row)':>14}  {'Sig pairs':>10}")
    print("  " + "─" * 50)
    for j in joints_test:
        row = z_matrix.loc[j].drop(index=j)
        mean_z = float(np.nanmean(row))
        n_sig_j = int(sig_matrix.loc[j].drop(index=j).sum())
        print(f"  {j:<22} {mean_z:>14.2f}  {n_sig_j:>10}")

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
        cbar_kws={"label": f"Z-score  ({freq_label})", "shrink": 0.6},
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

    ax.set_title(
        f"Joint-to-joint coherence matrix  ({freq_label})\n"
        f"Z-scores (permutation null)  |  Hatched = z ≤ {z_threshold:.2f} (non-significant)",
        fontsize=11, pad=28
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)
    save_figure(fig, PATH_FIGURES,
                f"fig{str(n).zfill(2)}_joint_coherence_matrix_{sequence_measure}"
                f"{trailing_file_name}.png",
                **SAVE_KW)
    plt.close(fig)


def power_spectrum_analysis(n, df, sequence_measure, generate_figure=True, **kwargs):
    from krajjat import analysis_functions as afunc

    if USE_REREF:
        trailing_file_name = "_rWaistBack"
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
        joint_layout="qualisys",
        n_jobs=-1,
        random_seed=42,
        show=False,
        path_save=path_png,
        verbosity=1,
    )
    params.update(kwargs)

    out = afunc.power_spectrum(df_ps, **params)
    out.save(path_json)

def _load_psd_file(kinematic_measure, acoustic_measure, use_reref=True, use_filtered=True):
    # Load data
    print("Loading dataframe...")
    if use_filtered:
        f = "filtered"
    else:
        f = "unfiltered"

    if use_reref:
        path = op.join(PATH_DATAFRAME, f"FS{int(FS)} {f}", f"qualisys_{kinematic_measure}_{acoustic_measure}_rWaistBack.gzip")
    else:
        path = op.join(PATH_DATAFRAME, f"FS{int(FS)} {f}", f"qualisys_{kinematic_measure}_{acoustic_measure}.gzip")
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
                if np.any(np.isnan(values)):
                    print(f"    {subj} / {trial}: {np.isnan(values).mean():.1%} NaN in signal, dropping trial for {joint}")
                    continue
                freqs, psd = welch(values, fs=fs, nperseg=nperseg)
                trial_psds.append(psd)
            if not trial_psds:
                continue
            mean_psd = np.mean(trial_psds, axis=0)
            if np.any(np.isnan(mean_psd)):
                print(f"    {subj} / {joint}: NaN in mean PSD after averaging, skipping")
                continue
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
            psds[joint][subj] = _normalize(mean_psd, sd_psd, freqs, normalization, global_max, subject_max,
                                           band_max_range)
            if psd_unit == "dB":
                mean_psd, sd_psd = psds[joint][subj]
                psds[joint][subj] = (10 * np.log10(mean_psd + 1e-30),
                                     10 * np.log10(sd_psd + 1e-30) if sd_psd is not None else None)

    return psds, freqs


def _stats_joints(psds, freqs, path_png_silhouette, show):
    from scipy.stats import friedmanchisquare

    print_section_title("STATS — JOINT POWER")

    joint_list = [j for j in psds if j != "Audio"]
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
                    font_size=0.5,
                    verbosity=0)


def _stats_subjects(psds, freqs, z_threshold=2.0, freq_bands=None):
    print_section_title("STATS — SUBJECT VARIABILITY")

    joint_list = [j for j in psds if j != "Audio"]
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

    # Mean pairwise PSD correlation per subject
    print("\nMean pairwise PSD correlation per subject (lower = more divergent):")
    from scipy.stats import spearmanr
    subj_psds = {s: np.concatenate([psds[j][s][0] for j in joint_list if s in psds[j]]) for s in subj_list}
    mean_rhos = {}
    for s in subj_list:
        corrs = [spearmanr(subj_psds[s], subj_psds[s2]).statistic
                 for s2 in subj_list if s2 != s]
        mean_rhos[s] = np.mean(corrs)
        print(f"  {s}: mean ρ = {mean_rhos[s]:.3f}")
    print(f"\n  Overall mean ρ: {np.mean(list(mean_rhos.values())):.3f}")

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


def _get_plot_dictionary(psds, freqs, show_average, freq_log, show_sd, include_audio):
    if freq_log:
        mask = freqs > 0
        freqs = freqs[mask]
        psds = {joint: {subj: (mean[mask], sd[mask] if sd is not None else None)
                        for subj, (mean, sd) in subj_psds.items()}
                for joint, subj_psds in psds.items()}

    plot_dictionary = {}
    for joint, subj_psds in psds.items():
        graph = Graph()

        if joint == "Audio" and not include_audio:
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


def _get_cache_path(kinematic_measure, acoustic_measure, use_reref, use_filtered, subjects, trials, joints,
                    nperseg, normalization, band_max_range, psd_unit, cache_dir="psd_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    key = str((kinematic_measure, acoustic_measure, use_reref, use_filtered, subjects, trials, joints,
               nperseg, normalization, band_max_range, psd_unit))
    hash_str = hashlib.md5(key.encode()).hexdigest()[:12]
    return op.join(cache_dir, f"psd_{hash_str}.pkl")


def _compute_subject_peak_freqs(psds, freqs, freq_range=None):
    """Per-joint, per-subject peak frequency (each subject's own PSD, not the group average)."""
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    else:
        mask = np.ones_like(freqs, dtype=bool)
    masked_freqs = freqs[mask]

    subject_peaks = {}  # joint -> {subj: peak_freq}
    for joint, subj_psds in psds.items():
        if joint == "Audio" or not subj_psds:
            continue
        subject_peaks[joint] = {}
        for subj, (mean_psd, sd_psd) in subj_psds.items():
            masked_psd = mean_psd[mask]
            if np.all(np.isnan(masked_psd)):
                continue
            subject_peaks[joint][subj] = masked_freqs[np.nanargmax(masked_psd)]
    return subject_peaks

def _compute_peak_stats(psds, freqs, freq_range=None, sd_line_height=0.8, ylim=None):
    """Per-joint peak frequency statistics: mean and SD across subjects' individual peaks, plus a
    y-position for the horizontal SD line fixed at `sd_line_height` (fraction of the y-axis range,
    0-1) — same absolute height on every joint's subplot. The y-range is computed only within
    `freq_range` (should match your plotted xlim), matching what plot_body_graphs actually displays."""
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    else:
        mask = np.ones_like(freqs, dtype=bool)
    masked_freqs = freqs[mask]

    if ylim is not None:
        y_min, y_max = ylim
    else:
        # Restrict to the same frequency window as the plot, not the full spectrum
        all_means_masked = [mean[mask] for joint, subj_psds in psds.items() if joint != "Audio"
                            for mean, _ in subj_psds.values()]
        y_min = float(np.nanmin([np.nanmin(m) for m in all_means_masked]))
        y_max = float(np.nanmax([np.nanmax(m) for m in all_means_masked]))
    y_fixed = y_min + sd_line_height * (y_max - y_min)

    peak_stats = {}
    for joint, subj_psds in psds.items():
        if joint == "Audio" or not subj_psds:
            continue
        subj_peak_freqs = []
        for subj, (mean_psd, sd_psd) in subj_psds.items():
            masked_psd = mean_psd[mask]
            if np.all(np.isnan(masked_psd)):
                continue
            subj_peak_freqs.append(masked_freqs[np.nanargmax(masked_psd)])
        if not subj_peak_freqs:
            continue

        mean_freq = np.mean(subj_peak_freqs)
        sd_freq = np.std(subj_peak_freqs) if len(subj_peak_freqs) > 1 else 0.0

        peak_stats[joint] = {"mean_freq": mean_freq, "sd_freq": sd_freq, "y_at_peak": y_fixed}
    return peak_stats


def _build_peak_overlay_lines(peak_stats, color="#202020", width=1.2, style="--",
                              sd_color=None, sd_width=1.5):
    """Vertical dashed line at the mean peak frequency, plus a horizontal segment spanning
    ± 1 SD, crossing it at the average curve's height — for use as `overlay_lines`."""
    if sd_color is None:
        sd_color = color
    overlay = {}
    for joint, stats in peak_stats.items():
        freq, sd, y = stats["mean_freq"], stats["sd_freq"], stats["y_at_peak"]
        vertical = GraphPlot(x=[freq, freq], y=[-1e6, 1e6], sd=None,
                             line_width=width, line_style=style, color=color, label=None)
        horizontal = GraphPlot(x=[freq - sd, freq + sd], y=[y, y], sd=None,
                               line_width=sd_width, line_style="-", color=sd_color, label=None)
        overlay[joint] = [vertical, horizontal]
    return overlay


def _build_peak_freq_silhouette_dict(subject_peaks, sd_threshold=None, ignore_sd_threshold=False):
    """Returns (sd_dict, mean_freq_dict, raw_sd_dict): sd_dict drives circle color, mean_freq_dict
    drives the displayed text, raw_sd_dict is unmasked for the console printout regardless of threshold."""
    sd_dictionary = {}
    mean_dictionary = {}
    raw_sd_dictionary = {}
    for joint, peaks_by_subj in subject_peaks.items():
        values = np.array(list(peaks_by_subj.values()))
        if len(values) == 0:
            continue
        mean_val = np.mean(values)
        sd_val = np.std(values) if len(values) > 1 else 0.0
        raw_sd_dictionary[joint] = sd_val
        if not ignore_sd_threshold and sd_threshold is not None and sd_val > sd_threshold:
            sd_dictionary[joint] = np.nan
            mean_dictionary[joint] = np.nan
        else:
            sd_dictionary[joint] = sd_val
            mean_dictionary[joint] = mean_val
    return sd_dictionary, mean_dictionary, raw_sd_dictionary


def plot_peak_freq_silhouette(n, kinematic_measure, acoustic_measure, df=None, use_reref=True, use_filtered=True,
                              subjects=None, trials=None, joints=None, force=False,
                              nperseg=256, normalization="fooof", band_max_range=(0.5, 4.0), psd_unit="default",
                              peak_freq_range=None, sd_threshold=None, ignore_sd_threshold=False,
                              min_scale=None, max_scale=None, path_png=None, show=True, **kwargs):

    print_section_title(f"{n}. PEAK FREQUENCY SILHOUETTE — {kinematic_measure.upper()} / {acoustic_measure.upper()}")
    if df is None:
        df = _load_psd_file(kinematic_measure, acoustic_measure, use_reref, use_filtered)
    df = _filter_df(df, subjects, trials, joints)
    fs = _infer_fs(df)
    cache_path = _get_cache_path(kinematic_measure, acoustic_measure, use_reref, use_filtered, subjects, trials,
                                 joints, nperseg, normalization, band_max_range, psd_unit)

    if op.exists(cache_path) and not force:
        print(f"Loading cached PSDs from {cache_path}...")
        with open(cache_path, "rb") as f:
            psds, freqs = pickle.load(f)
    else:
        psds, freqs = _get_psds(df, fs, nperseg, normalization, band_max_range, psd_unit)
        with open(cache_path, "wb") as f:
            pickle.dump((psds, freqs), f)
        print(f"PSDs cached to {cache_path}")

    subject_peaks = _compute_subject_peak_freqs(psds, freqs, peak_freq_range)
    sd_dict, mean_dict, raw_sd_dict = _build_peak_freq_silhouette_dict(subject_peaks, sd_threshold,
                                                                       ignore_sd_threshold)

    print("\n  Peak frequency per joint (mean ± SD across subjects):")
    for joint in sorted(subject_peaks.keys()):
        values = list(subject_peaks[joint].values())
        mean_val = np.mean(values)
        sd_val = raw_sd_dict.get(joint, 0.0)
        excluded = " [excluded: SD too high]" if np.isnan(sd_dict.get(joint, np.nan)) else ""
        print(f"    {joint:<20} {mean_val:.2f} ± {sd_val:.2f} Hz (n={len(values)}){excluded}")

    valid_vals = [v for v in sd_dict.values() if not np.isnan(v)]
    auto_min, auto_max = (min(valid_vals), max(valid_vals)) if valid_vals else (0, 1)
    min_scale = auto_min if min_scale is None else min_scale
    max_scale = auto_max if max_scale is None else max_scale

    plot_silhouette(sd_dict, display_dictionary=mean_dict, joint_layout="qualisys",
                    title_scale="SD of peak frequency (Hz)",
                    min_scale=min_scale, max_scale=max_scale, show_scale=True,
                    show_values=True, value_format=".1f",
                    show=show, path_save=path_png, **kwargs)

    return sd_dict, mean_dict, raw_sd_dict

def plot_psd(n, kinematic_measure, acoustic_measure, df=None, use_reref=True, use_filtered=True,
             subjects=None, trials=None, joints=None, force=False,
             nperseg=256, normalization=None, band_max_range=(0.5, 4.0),
             psd_unit="default", freq_log=False, show_stats=True, include_audio=True,
             show_average=True, show_sd=False, show=True, path_png=None, path_png_silhouette=None,
             mark_peak_freq=False, peak_freq_range=None, sd_line_height=0.8, ylim=None,
             peak_line_color="#202020", peak_line_width=1.2, peak_line_style="--",
             exclude_curves=None, exclude_note=None, **kwargs):

    print_section_title(f"{n}. PSD DIAGNOSTIC — {kinematic_measure.upper()} / {acoustic_measure.upper()}")
    if df is None:
        df = _load_psd_file(kinematic_measure, acoustic_measure, use_reref, use_filtered)
    df = _filter_df(df, subjects, trials, joints)
    fs = _infer_fs(df)
    cache_path = _get_cache_path(kinematic_measure, acoustic_measure, use_reref, use_filtered, subjects, trials,
                                 joints, nperseg, normalization, band_max_range, psd_unit)

    if op.exists(cache_path) and force == False:
        print(f"Loading cached PSDs from {cache_path}...")
        with open(cache_path, "rb") as f:
            psds, freqs = pickle.load(f)
    else:
        psds, freqs = _get_psds(df, fs, nperseg, normalization, band_max_range, psd_unit)
        with open(cache_path, "wb") as f:
            pickle.dump((psds, freqs), f)
        print(f"PSDs cached to {cache_path}")

    # Remove specific (joint, subject) curves flagged as capture artifacts (e.g. marker flicker).
    # Applied AFTER caching, so the cache keeps the full data — exclusion is a display-time choice.
    if exclude_curves:
        for joint, subjs_to_exclude in exclude_curves.items():
            if joint not in psds:
                continue
            for subj in subjs_to_exclude:
                if subj in psds[joint]:
                    del psds[joint][subj]
                    print(f"  Excluded {subj} from {joint} (flagged as artifact)")

    if show_stats:
        _stats_joints(psds, freqs, path_png_silhouette, show)
        _stats_subjects(psds, freqs)

    if mark_peak_freq:
        if "overlay_lines" in kwargs:
            raise ValueError("mark_peak_freq=True builds its own overlay_lines; "
                            "passing overlay_lines manually at the same time isn't supported yet.")
        peak_stats = _compute_peak_stats(psds, freqs, peak_freq_range, sd_line_height, ylim)
        kwargs["overlay_lines"] = _build_peak_overlay_lines(peak_stats, peak_line_color, peak_line_width,
                                                             peak_line_style)

    plot_dictionary = _get_plot_dictionary(psds, freqs, show_average, freq_log, show_sd, include_audio)

    # Build without saving/showing yet, so the exclusion note can be added first
    fig = plot_body_graphs(plot_dictionary, x_scale_log=freq_log, show=False, path_save=None,
                           close_fig=False, **kwargs)

    # if exclude_curves:
    #     if exclude_note is None:
    #         n_excluded = sum(len(v) for v in exclude_curves.values())
    #         joints_str = ", ".join(sorted(exclude_curves.keys()))
    #         exclude_note = f"{n_excluded} curve(s) excluded from {joints_str} (capture artifact)"
    #     fig.text(0.5, 0.005, exclude_note, ha="center", va="bottom", fontsize=8,
    #              style="italic", color="#606060")

    if path_png is not None:
        folder = op.split(path_png)[0]
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        fig.savefig(path_png)
    if show:
        plt.show()
    plt.close(fig)

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
        subj_short = subj
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
            selected_components=list(range(n_components)),  # ← add this
            show=False,
            path_save=None,
            verbosity=0,
        )

        # Align signs: flip each component so that the joint with the largest
        # absolute mean loading across subjects is always positive.
        # Reference joint per PC = the one with highest |loading| in this subject.
        # We align to HandLeft if present, otherwise to the top-loading joint.
        reference_joint = "HandInLeft" if "HandInLeft" in pca_comps[0] else None
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
                for s in [subj for subj in subjects]
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
            title_silhouette=[s for s in subjects],
            title_scale="|Loading|",
            min_scale=0,
            max_scale=vmax,
            color_scheme="celsius",
            show_values=False,
            value_format=".2f",
            font_size=0.5,
            n_cols=6,
            show=False,
            path_save=op.join(PATH_FIGURES,
                              f"fig{str(n).zfill(2)}a_pca_pc{pc_idx+1}_loadings.png"),
            resolution=(1200, 1080),
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

def quantity_of_movement_by_joint(n, df, fs, generate_figure=True, path_save=None):
    """
    Quantity of movement (QoM) per joint, as a single overall silhouette.

    QoM is the integral of speed over time (total distance traveled), not a
    mean -- computed per (subject, trial, joint) as sum(speed) * dt, then
    averaged across trials within each subject, then averaged across
    subjects. This two-step averaging (trial -> subject -> group) ensures
    subjects with more trials/frames don't dominate the group average, the
    same way mean_speed_by_joint() treats every subject equally regardless
    of how many frames they contributed.

    df: velocity dataframe with columns [subject, trial, label, value]
        (value = instantaneous speed, one row per frame).
    fs: sampling rate (Hz), used to convert per-frame speed into distance.
    """
    print_section_title(f"{n}. QUANTITY OF MOVEMENT BY JOINT")

    dt = 1 / fs
    joints = list(df["label"].unique())

    # sum(speed) * dt per (subject, trial, joint) -> per-trial distance traveled
    qom_per_trial = (df.groupby(["subject", "trial", "label"])["value"]
                       .sum().mul(dt).unstack("label").reindex(columns=joints))

    # average across trials within each subject, then across subjects
    qom_per_subject = qom_per_trial.groupby("subject").mean()
    qom_overall = qom_per_subject.mean()

    print("\nOverall quantity of movement per joint (cm, averaged trial -> subject -> group):")
    print((qom_overall * 100).round(2).to_string())

    if generate_figure:
        plot_dict_overall = {joint: float(qom_overall[joint] * 100) for joint in joints}
        filename = f"fig{str(n).zfill(2)}_quantity_of_movement_by_joint.png"
        out_dir = path_save or PATH_FIGURES
        plot_silhouette(plot_dict_overall,
                        title_scale="Quantity of movement (cm)",
                        min_scale=0, max_scale=float(qom_overall.max() * 100),
                        color_scheme="celsius",
                        path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"),
                        show=False,
                        path_save=op.join(out_dir, filename),
                        resolution=(600, 1080),
                        show_values=True,
                        value_format=".1f",
                        font_size=0.5,
                        verbosity=0)

    return qom_overall, qom_per_subject


def quantity_of_movement_by_joint_by_subject(n, df, fs, generate_figure=True):
    """
    Quantity of movement (QoM) per joint per subject, as a grid of
    silhouettes (one per subject).

    Same trial -> subject averaging as quantity_of_movement_by_joint(), but
    the group-level averaging step is skipped so that each subject's own
    average QoM per joint is shown individually.
    """
    print_section_title(f"{n}. QUANTITY OF MOVEMENT BY JOINT BY SUBJECT")

    dt = 1 / fs
    joints = list(df["label"].unique())

    qom_per_trial = (df.groupby(["subject", "trial", "label"])["value"]
                       .sum().mul(dt).unstack("label").reindex(columns=joints))
    qom_per_subject = qom_per_trial.groupby("subject").mean()

    print("\nQuantity of movement per joint per subject (cm):")
    print((qom_per_subject * 100).round(2).to_csv(sep="\t"))

    if generate_figure:
        plot_dict_subj = {joint: (qom_per_subject[joint] * 100).values for joint in joints}
        n_subj = len(subjects)
        n_cols_grid = 6
        plot_silhouette(plot_dict_subj,
                        title="Quantity of movement per joint per subject",
                        title_silhouette=[s for s in subjects],
                        title_scale="Quantity of movement (cm)",
                        min_scale=0, max_scale=float((qom_per_subject * 100).max().max()),
                        n_cols=n_cols_grid,
                        show=False,
                        color_scheme="celsius",
                        path_save=op.join(PATH_FIGURES,
                                          f"fig{str(n).zfill(2)}_quantity_of_movement_by_joint_by_subject.png"),
                        resolution=(1200, 1080),
                        display_ratio=0.95,
                        verbosity=0)

    return qom_per_subject

if __name__ == "__main__":
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

    subjects_minus_s004 = ["S" + str(i).zfill(3) for i in range(1, 4)] + ["S" + str(i).zfill(3) for i in range(5, 25)]

    n_subjects = len(subjects)

    print(f"Joints    : {joints}")
    print(f"Subjects  : {subjects}")
    print(f"Conditions: {conditions}")

    # mean_speed_by_joint(1, df_mocap, generate_figures)
    # mean_speed_comparison_between_joints(2, df_mocap, generate_figures)
    # mean_speed_by_subject(3, df_mocap, generate_figures)
    # mean_speed_by_subject_joint(4, df_mocap, "HandInRight", generate_figures)
    # mean_speed_by_subject_joint(4, df_mocap, "HandInLeft", generate_figures)
    # mean_speed_by_joint_by_subject(4, df_mocap, generate_figures)
    # mean_speed_by_joint_by_subject_normalized(5, df_mocap, generate_figures)
    # mean_speed_deviation_by_joint_by_subject(6, df_mocap, generate_figures)
    # joint_rank(7, df_mocap, generate_figures)
    # joint_rank_stability(8, df_mocap, generate_figures)
    # joint_rank_by_subject(9, df_mocap, generate_figures)
    # trial_order(10, df_mocap, None, generate_figures)
    # trial_order_by_subject(11, df_mocap, None, generate_figures)

    # Trial order (speed) across the three reported joint-scopes -- each call in
    # trial_order() is fully independent, so correct the family of 3 here rather
    # than inside the function (matches the "correct across the reported sub-
    # analyses" convention used throughout this pipeline).
    # trial_order_scopes = [("all joints", None, 12), ("right hand", "HandInRight", 13),
    #                       ("left hand", "HandInLeft", 14)]
    # trial_order_results = []
    # for label, joint_arg, fig_n in trial_order_scopes:
    #     r_all, p_pearson, p_lmm = trial_order(fig_n, df_mocap, joint_arg, generate_figures)
    #     trial_order_results.append((label, r_all, p_pearson, p_lmm))
    #
    # pearson_ps = [r[2] for r in trial_order_results]
    # lmm_ps = [r[3] for r in trial_order_results]
    # _, pearson_ps_fdr, _, _ = multipletests(pearson_ps, method="fdr_bh")
    # _, lmm_ps_fdr, _, _ = multipletests(lmm_ps, method="fdr_bh")
    #
    # print("\nTrial order (speed) across joint-scopes -- FDR-corrected (family = these 3 scopes)")
    # print(f"  {'Scope':<12} {'r':>7} {'p (Pearson)':>12} {'p_FDR (Pearson)':>16} {'p (LMM)':>9} {'p_FDR (LMM)':>13}")
    # for (label, r_all, p_pearson, p_lmm), p_pearson_fdr, p_lmm_fdr in zip(trial_order_results, pearson_ps_fdr,
    #                                                                       lmm_ps_fdr):
    #     print(f"  {label:<12} {r_all:>7.3f} {p_pearson:>12.4f} {p_pearson_fdr:>16.4f} {p_lmm:>9.4f} "
    #           f"{p_lmm_fdr:>13.4f}")

    # trial_order_by_subject(13, df_mocap, ["HandInRight", "HandOutRight"], generate_figures)
    # mean_speed_by_cartoon(14, df_mocap, None, generate_figures)
    # mean_speed_by_show(15, df_mocap, None, generate_figures)
    # mean_speed_by_cartoon(16, df_mocap, None, generate_figures)
    # mean_speed_by_cartoon(16, df_mocap, "HandInRight", generate_figures)
    # mean_speed_by_cartoon(16, df_mocap, "HandInLeft", generate_figures)
    # mean_speed_by_show(17, df_mocap, ["HandInRight", "HandOutRight"], generate_figures)
    # bilateral_symmetry(18, df_mocap, generate_figures)
    # joint_correlation_matrix(19, df_mocap, generate_figures)
    # joint_correlation(20, df_mocap, "HandInRight", generate_figures)
    # joint_correlation(21, df_mocap, "HeadFront", generate_figures)
    # subjects_to_consider = ["S001", "S002", "S003", "S005", "S006", "S007", "S008", "S009", "S010", "S011", "S012",
    #                         "S013", "S014", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S022", "S023", "S024"]
    subjects_to_consider = ["S001", "S002", "S003", "S005", "S006", "S007", "S008", "S009", "S010", "S011", "S012",
                            "S014", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S022", "S024"]
    # joint_coherence_body_graph(22, "HandInRight", "velocity", subjects_to_consider, "average", generate_figures)
    # joint_coherence_silhouette(22, "HandInRight", "velocity", subjects_to_consider, [0.2], "average", generate_figures)
    # joint_coherence_body_graph(23, "HandInLeft", "velocity", subjects_to_consider, "average", generate_figures)
    # joint_coherence_silhouette(23, "HandInLeft", "velocity", subjects_to_consider, [0.2], "average", generate_figures)
    # joint_coherence_body_graph(24, "HeadFront", "velocity", subjects_to_consider, "average", generate_figures)
    # joint_coherence_silhouette(24, "HeadFront", "velocity", subjects_to_consider, [0.2], "average", generate_figures)
    # joint_coherence_matrix(25, "velocity", 0.5, generate_figures)
    # joint_coherence_matrix(26, "velocity", 1, generate_figures)
    # power_spectrum_analysis(27, df_mocap, "velocity", generate_figures,
    #                         average="subjects",
    #                         series=None)
    # power_spectrum_analysis(28, df_mocap, "velocity", generate_figures,
    #                         average="subjects",
    #                         series=None,
    #                         result_type="z-scores",
    #                         permutation_method="shift",
    #                         number_of_randperms=1000,
    #                         fit_background=False,
    #                         signif_style="threshold")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False,
                     "font.family": "Metropolis", "xtick.labelsize": 8, "ytick.labelsize": 8}
    sns_kwargs = {"style": "whitegrid", "font_scale": 1.2, "rc": custom_params}
    mpl_kwargs = {"hspace": 1, "left": 0.07, "right": 0.93, "top": 0.93, "bottom": 0.07}
    kinematic_measure = "acceleration"
    normalization = "fooof"

    for kinematic_measure in ["velocity", "acceleration"]:
        for normalization in ["subject_peak", "fooof"]:
            plot_psd(29, kinematic_measure, "envelope", use_reref=True,
                     subjects = ["S001", "S002", "S003", "S005", "S006", "S007", "S008", "S009", "S010", "S011", "S012",
                                 "S014", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S022", "S024"],
                     # subjects = "S016",
                     trials = None, xlim = [0, 10], legend = True, show_average = True,
                     joints = None, joint_layout = "auto", include_audio=False,
                     nperseg=256, normalization=normalization,
                     x_label="Frequency (Hz)", y_label="Normalized amplitude",
                     sns_kwargs=sns_kwargs,
                     mpl_kwargs=mpl_kwargs, show=False, freq_log=False,
                     path_png=op.join(PATH_FIGURES, f"qualisys_psd_{kinematic_measure}_{normalization}.png"))

    # plot_psd(30, "velocity",
    #          "envelope",
    #          ["HeadRight", "HeadLeft", "HandInRight", "HandInLeft", "WaistBackRight", "WaistBackLeft", "ToetipRight", "ToetipLeft"],
    #          generate_figure=generate_figures)
    # pca_analysis(31, df_mocap, "velocity", n_components=3, generate_figure=generate_figures)

    # generate_figures = True
    # i = 101
    # frequency = [0.2]
    #
    # joints = ["HandInRight", "HandInLeft", "HeadFront"]
    # value_types = ["raw", "z-score"]
    # levels = ["subject", "group"]
    #
    # for joint, value_type, level in product(joints, value_types, levels):
    #     if value_type == "raw":
    #         joint_coherence_body_graph(i, joint, "velocity", value_type, level, None, generate_figures, subjects_to_consider)
    #         joint_coherence_silhouette(i + 1, joint, "velocity", frequency, value_type, level, generate_figures)
    #     i += 2

    # quantity_of_movement_by_joint(42, df_mocap, FS, generate_figures)
    # quantity_of_movement_by_joint_by_subject(43, df_mocap, FS, generate_figures)