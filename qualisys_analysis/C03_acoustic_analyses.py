import math
import os.path as op

from krajjat import Graph
from scipy.signal import welch
from statsmodels.formula.api import mixedlm

from constants import *
import pandas as pd
import numpy as np
from helper_functions import *
from scipy.stats import kruskal, spearmanr, linregress, friedmanchisquare, wilcoxon
from colorama import Fore, Style
from itertools import combinations
from scipy.stats import mannwhitneyu, chi2_contingency, norm, ttest_1samp
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.colors as mcolors

PATH_FIGURES = op.join(PATH_PLOTS, "Acoustics")
os.makedirs(PATH_FIGURES, exist_ok=True)

def darken(color, factor=0.6):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))

def lighten(hex_color, factor=0.65):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * factor)

def plot_acoustic_measure_per_participant(n, acoustic_measure, filtered="unfiltered", generate_figure=True):
    print_section_title(f"{n}. Acoustic measure per participant: {acoustic_measure.title()}")

    df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}", f"{SYSTEM.lower()}_velocity_{acoustic_measure}.gzip"))
    df = df[df["measure"] == acoustic_measure].copy()

    # Exclude silent frames encoded as 0 or below
    df = df[df["value"] >= 1e-5]

    if acoustic_measure == "pitch":
        df = df[df["value"] >= 75]

    subjects = df["subject"].unique()
    frame_data = {s: df[df["subject"] == s]["value"].values for s in subjects}
    subjects = sorted(df["subject"].unique(), key=lambda s: np.mean(frame_data[s]))
    n_subjects = len(subjects)

    # Descriptive table
    print("\t".join(["Subject", "N frames", "Mean", "SD", "Median", "IQR", "Min", "Max"]))
    for s in subjects:
        v = frame_data[s]
        q25, q75 = np.percentile(v, [25, 75])
        short = s.split("_", 1)[-1] if "_" in s else s
        print("\t".join([short, str(len(v)), f"{np.mean(v):.3f}", f"{np.std(v):.3f}", f"{np.median(v):.3f}",
                         f"{q75 - q25:.3f}", f"{np.min(v):.3f}", f"{np.max(v):.3f}"]))

    # Subsampled arrays for stats
    sampling_rate = FS
    step = max(1, sampling_rate)
    sub_data = {s: frame_data[s][::step] for s in subjects}
    sub_arrays = [sub_data[s] for s in subjects]

    n_sub = {s: len(sub_data[s]) for s in subjects}
    print(f"\n  Subsampled N frames for stats: "
          + ", ".join(f"{s.split('_',1)[-1]}={n_sub[s]}" for s in subjects))

    # Kruskal-Wallis
    print(f"\n  Kruskal-Wallis (subsampled frames, between participants):")
    kw_stat, kw_p = kruskal(*sub_arrays)

    N_total = sum(len(a) for a in sub_arrays)
    k = n_subjects
    eta_sq = max(0.0, (kw_stat - k + 1) / (N_total - k)) if N_total > k else np.nan

    sig = (Fore.GREEN + "✓" + Style.RESET_ALL) if kw_p < 0.05 \
          else (Fore.RED + "✗" + Style.RESET_ALL)
    p_str = f"{kw_p:.4f}" if kw_p >= 0.001 else "< .001"
    print(f"  {sig}  H({k - 1}) = {kw_stat:.3f}, p = {p_str}, η² = {eta_sq:.3f}")

    pairs = list(combinations(range(n_subjects), 2))
    pair_labels = [(subjects[i], subjects[j]) for i, j in pairs]
    raw_pvals = []

    for i, j in pairs:
        _, p = mannwhitneyu(sub_arrays[i], sub_arrays[j], alternative="two-sided")
        raw_pvals.append(p)

    reject, pvals_corr, _, _ = multipletests(raw_pvals, method="holm")

    # print(f"\n  Pairwise Mann-Whitney U (Holm-corrected, subsampled):")
    # for (s1, s2), p_raw, p_corr, rej in zip(pair_labels, raw_pvals, pvals_corr, reject):
    #     short1 = s1.split("_", 1)[-1] if "_" in s1 else s1
    #     short2 = s2.split("_", 1)[-1] if "_" in s2 else s2
    #     sym = (Fore.GREEN + "✓" + Style.RESET_ALL) if rej \
    #           else (Fore.RED + "✗" + Style.RESET_ALL)
    #     print(f"  {sym}  {short1} vs {short2:<16}  "
    #           f"p_raw = {p_raw:.4f},  p_corr = {p_corr:.4f}")

    all_subjects = sorted(df["subject"].unique())  # fixed alphabetical order
    colors = plt.cm.tab20.colors
    subject_colors = {s: colors[i % len(colors)] for i, s in enumerate(all_subjects)}

    # Figure
    if generate_figure:

        fig, ax = plt.subplots(figsize=(max(10, n_subjects * 0.9), 5))  # tighter per-subject width
        parts = ax.violinplot([frame_data[s] for s in subjects], positions=range(n_subjects),
                              showmeans=False, showmedians=False, showextrema=False,
                              widths=0.55)

        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(subject_colors[subjects[i]])
            pc.set_alpha(0.55)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.8)
            # Clip KDE tails that bleed below zero
            for path in pc.get_paths():
                path.vertices[:, 1] = np.clip(path.vertices[:, 1], 0, None)

        # Mean (diamond) and median (horizontal line) markers
        for i, s in enumerate(subjects):
            v = frame_data[s]
            q25, q75 = np.percentile(v, [25, 75])

            # IQR bar
            ax.vlines(i, q25, q75, color=subject_colors[s],
                      linewidth=3, zorder=3, alpha=0.9)

            # Median line
            ax.hlines(np.median(v), i - 0.08, i + 0.08,
                      color="black", linewidth=2, zorder=4)

            # Mean diamond
            ax.scatter(i, np.mean(v),
                       color="white", edgecolor="black",
                       linewidth=1.5, s=80, zorder=5, marker="D")

        short_labels = [s.split("_", 1)[-1] if "_" in s else s for s in subjects]
        ax.set_xticks(range(n_subjects))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10)

        ax.set_title(f"{acoustic_measure.capitalize()} distribution per participant", fontsize=12,
                     fontweight="bold")
        ax.set_ylabel(acoustic_measure.capitalize(), fontsize=11)
        ax.set_xlabel("Participant", fontsize=11)

        # KW annotation
        kw_text = f"KW: H({k - 1}) = {kw_stat:.2f}, p = {p_str}, η² = {eta_sq:.2f}"
        ax.text(0.99, 0.97, kw_text, transform=ax.transAxes,
                fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            plt.scatter([], [], marker="D", color="white", edgecolor="black",
                        linewidth=1.5, s=80, label="Mean"),
            Line2D([0], [0], color="black", linewidth=2, label="Median"),
            Line2D([0], [0], color="gray", linewidth=3, label="IQR"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2) }_{acoustic_measure}_per_participant.png", **SAVE_KW)
        plt.show()

def voiced_unvoiced_ratio(n, filtered="unfiltered", generate_figure=True):
    print_section_title(f"{n}. VOICED/UNVOICED RATIO")

    df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}", f"{SYSTEM.lower()}_velocity_pitch.gzip"))
    df = df[df["measure"] == "pitch"].copy()

    subjects = sorted(df["subject"].unique())
    short_labels = [s.split("_", 1)[-1] if "_" in s else s for s in subjects]

    voiced_pct = []
    print(f"\n{'Subject':<16} {'Total frames':>14} {'Voiced':>10} {'Unvoiced':>10} {'Voiced %':>10}")
    print("  " + "─" * 55)
    for s in subjects:
        sub = df[df["subject"] == s]
        total = len(sub)
        voiced = (sub["value"] >= 75).sum()
        unvoiced = total - voiced
        pct = 100 * voiced / total
        voiced_pct.append(pct)
        short = s.split("_", 1)[-1] if "_" in s else s
        print(f"  {short:<14} {total:>14} {voiced:>10} {unvoiced:>10} {pct:>9.1f}%")

    # Build contingency table: rows = subjects, cols = [voiced, unvoiced]
    counts = []
    for s in subjects:
        sub = df[df["subject"] == s]
        total = len(sub)
        voiced = int((sub["value"] >= 75).sum())
        counts.append([voiced, total - voiced])

    counts_arr = np.array(counts)
    chi2, p_chi2, dof, _ = chi2_contingency(counts_arr)

    sorted_indices = np.argsort(voiced_pct)
    subjects = [subjects[i] for i in sorted_indices]
    short_labels = [short_labels[i] for i in sorted_indices]
    voiced_pct = [voiced_pct[i] for i in sorted_indices]
    counts_arr = counts_arr[sorted_indices]  # keep aligned for pairwise tests

    # Effect size: Cramér's V
    n_total = counts_arr.sum()
    cramers_v = np.sqrt(chi2 / (n_total * (min(len(subjects), 2) - 1)))

    p_str = f"{p_chi2:.4f}" if p_chi2 >= 0.001 else "< .001"
    print(f"\n  Chi-square test of independence (voiced vs unvoiced counts):")
    sig = (Fore.GREEN + "✓" + Style.RESET_ALL) if p_chi2 < 0.05 else (Fore.RED + "✗" + Style.RESET_ALL)
    print(f"  {sig}  χ²({dof}) = {chi2:.3f}, p = {p_str}, Cramér's V = {cramers_v:.3f}")

    # Pairwise proportion z-tests (Holm-corrected)
    pairs = list(combinations(range(len(subjects)), 2))
    raw_pvals = []
    for i, j in pairs:
        n1, n2 = counts_arr[i].sum(), counts_arr[j].sum()
        p1, p2 = counts_arr[i, 0] / n1, counts_arr[j, 0] / n2
        p_pool = (counts_arr[i, 0] + counts_arr[j, 0]) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        raw_pvals.append(2 * (1 - norm.cdf(abs(z))))

    reject, pvals_corr, _, _ = multipletests(raw_pvals, method="holm")

    print(f"\n  Pairwise proportion z-tests (Holm-corrected):")
    for (i, j), p_raw, p_corr, rej in zip(pairs, raw_pvals, pvals_corr, reject):
        short1 = short_labels[i]
        short2 = short_labels[j]
        sym = (Fore.GREEN + "✓" + Style.RESET_ALL) if rej else (Fore.RED + "✗" + Style.RESET_ALL)
        p_raw_str = f"{p_raw:.4f}" if p_raw >= 0.0001 else "< .001"
        p_corr_str = f"{p_corr:.4f}" if p_corr >= 0.0001 else "< .001"
        print(f"  {sym}  {short1} vs {short2:<14}  "
              f"p_raw = {p_raw_str},  p_corr = {p_corr_str}")

    all_subjects = sorted(df["subject"].unique())  # fixed alphabetical order
    colors = plt.cm.tab20.colors
    subject_colors = {s: colors[i % len(colors)] for i, s in enumerate(all_subjects)}

    if generate_figure:
        colors = plt.cm.tab20.colors  # tab20 for 23 subjects
        n_subjects = len(subjects)

        fig, ax = plt.subplots(figsize=(max(10, n_subjects * 0.9), 5))
        bars = ax.bar(range(n_subjects), voiced_pct,
                      color=[subject_colors[s] for s in subjects],
                      edgecolor="black", linewidth=0.8, width=0.7, zorder=3)

        # Unvoiced portion on top, in a muted color
        unvoiced_pct = [100 - v for v in voiced_pct]
        ax.bar(range(n_subjects), unvoiced_pct, bottom=voiced_pct,
               color=[subject_colors[s] for s in subjects],
               edgecolor="black", linewidth=0.8, width=0.7,
               alpha=0.25, hatch="////", zorder=3)

        # Percentage labels inside bars
        for i, (v, u) in enumerate(zip(voiced_pct, unvoiced_pct)):
            ax.text(i, v / 2, f"{v:.1f}%", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", rotation=90)
            if u > 5:  # only label unvoiced if segment is tall enough
                ax.text(i, v + u / 2, f"{u:.1f}%", ha="center", va="center",
                        fontsize=7, color="#444", rotation=90)

        ax.set_xticks(range(n_subjects))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Percentage of frames (%)", fontsize=11)
        ax.set_xlabel("Participant", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_title("Voiced / unvoiced ratio per participant", fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="gray", edgecolor="black", linewidth=0.8, alpha=0.9, label="Voiced"),
            Patch(facecolor="gray", edgecolor="black", linewidth=0.8, alpha=0.25,
                  hatch="////", label="Unvoiced"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

        plt.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_voiced_unvoiced_ratio.png", **SAVE_KW)
        plt.close(fig)
        
def _get_voiced_mask(threshold=75):
    """Voicing determined from RAW (unfiltered) pitch — meaningful only on the absolute Hz scale.
    Returns a (subject, trial, timestamp, voiced) dataframe to merge against any other dataframe."""
    df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} unfiltered", f"{SYSTEM.lower()}_velocity_pitch.gzip"))
    df = df[df["measure"] == "pitch"][["subject", "trial", "timestamp", "value"]].copy()
    df["voiced"] = df["value"] >= threshold
    return df[["subject", "trial", "timestamp", "voiced"]]

def acoustic_measure_correlations(n, filtered="filtered", generate_figure=True, ax=None):
    print_section_title(f"{n}. CORRELATIONS BETWEEN ACOUSTIC MEASURES")

    def _no_lead_zero(val, decimals):
        """Format a float with `decimals` places, dropping the leading 0 before the decimal point."""
        s = f"{val:.{decimals}f}"
        if s.startswith("0."):
            return s[1:]
        elif s.startswith("-0."):
            return "-" + s[2:]
        return s

    measures = ["envelope", "intensity", "pitch", "f1", "f2"]

    # Load each measure's raw values (no absolute-scale thresholding here — voicing is handled separately below)
    dfs = {}
    for m in measures:
        df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}", f"{SYSTEM.lower()}_velocity_{m}.gzip"))
        df = df[df["measure"] == m][["subject", "trial", "timestamp", "value"]].copy()
        df = df.rename(columns={"value": m})
        dfs[m] = df

    # Merge on subject + trial + frame
    merged = dfs["envelope"]
    for m in measures[1:]:
        merged = merged.merge(dfs[m], on=["subject", "trial", "timestamp"], how="inner")

    # Restrict to voiced frames only, determined from raw (unfiltered) pitch — applies regardless of `filtered`
    voiced_mask = _get_voiced_mask(threshold=75)
    merged = merged.merge(voiced_mask, on=["subject", "trial", "timestamp"], how="inner")
    merged = merged[merged["voiced"]].drop(columns="voiced")

    print(f"\n  Total shared frames (all measures voiced/valid): {len(merged)}")

    subjects = sorted(merged["subject"].unique())
    pairs = list(combinations(measures, 2))

    # Fisher-averaged Spearman per pair
    r_matrix = pd.DataFrame(np.nan, index=measures, columns=measures)
    p_matrix = pd.DataFrame(np.nan, index=measures, columns=measures)

    raw_ps = []
    coords = []

    for m1, m2 in pairs:
        z_vals = []
        for s in subjects:
            sub = merged[merged["subject"] == s]
            if len(sub) < 10:
                continue
            r, _ = spearmanr(sub[m1], sub[m2])
            z_vals.append(np.arctanh(np.clip(r, -0.9999, 0.9999)))
        _, p = ttest_1samp(z_vals, 0)
        mean_z = np.mean(z_vals)
        r_avg = np.tanh(mean_z)
        r_matrix.loc[m1, m2] = r_avg
        r_matrix.loc[m2, m1] = r_avg
        raw_ps.append(p)
        coords.append((m1, m2))
        p_str = f"{p:.4f}" if p >= 0.001 else "< .001"
        print(f"  {m1} × {m2}: mean r = {r_avg:.3f}, p = {p_str}")

    # FDR correction across all pairs in this matrix
    _, p_corrected, _, _ = multipletests(raw_ps, method="fdr_bh")

    for (m1, m2), p in zip(coords, p_corrected):
        p_matrix.loc[m1, m2] = p
        p_matrix.loc[m2, m1] = p
        p_str = f"{p:.4f}" if p >= 0.001 else "< .001"
        print(f"  {m1} × {m2}: mean r = {r_matrix.loc[m1, m2]:.3f}, p (FDR-corrected) = {p_str}")

    for m in measures:
        r_matrix.loc[m, m] = 1.0
        p_matrix.loc[m, m] = np.nan

    if generate_figure:
        annot = pd.DataFrame("", index=measures, columns=measures)

        external_ax = ax is not None
        if not external_ax:
            fig, ax = plt.subplots(figsize=(7, 6))

        for m1 in measures:
            for m2 in measures:
                if m1 == m2:
                    annot.loc[m1, m2] = "—"
                else:
                    r_val = r_matrix.loc[m1, m2]
                    p_val = p_matrix.loc[m1, m2]
                    p_str = "< .001" if p_val < 0.001 else f"= {_no_lead_zero(p_val, 3)}"
                    annot.loc[m1, m2] = f"r = {_no_lead_zero(r_val, 2)}\np {p_str}"

        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.ticker import FuncFormatter
        cmap = LinearSegmentedColormap.from_list("my_cmap", ["#028090", "white", "#B82F2B"])

        sns.heatmap(
            r_matrix, ax=ax,
            cmap=cmap, vmin=-1, vmax=1, center=0,
            annot=annot, fmt="", annot_kws={"size": 9, "color": "#202020"},
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Mean Spearman r (Fisher-averaged)", "shrink": 0.8},
            xticklabels=[m.capitalize() for m in measures],
            yticklabels=[m.capitalize() for m in measures],
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: _no_lead_zero(val, 2)))

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", top=False)

        for i, m1 in enumerate(measures):
            for j, m2 in enumerate(measures):
                text_obj = ax.texts[i * len(measures) + j]
                if m1 != m2 and p_matrix.loc[m1, m2] < 0.05:
                    text_obj.set_fontweight("bold")

        if not external_ax:
            plt.tight_layout()
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_acoustic_correlations.png", **SAVE_KW)
            plt.close(fig)
            
def acoustic_measure_by_trial_order(n, acoustic_measure, filtered="unfiltered", generate_figure=True):
    print_section_title(f"{n}. {acoustic_measure.title()} by trial order")

    df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}", f"{SYSTEM.lower()}_velocity_{acoustic_measure}.gzip"))
    df = df[df["measure"] == acoustic_measure].copy()
    df = df[df["value"] >= 1e-5]
    if acoustic_measure == "pitch":
        df = df[df["value"] >= 75]

    trial_means = df.groupby(["subject", "trial"])["value"].mean().reset_index()
    trial_means = trial_means.rename(columns={"value": "mean_value"})
    trial_means["trial_order"] = trial_means.groupby("subject")["trial"].transform(
        lambda x: x.astype(str).map({v: i + 1 for i, v in enumerate(sorted(x.astype(str).unique()))})
    )

    # Overall Spearman
    r_all, p_all = spearmanr(trial_means["trial_order"], trial_means["mean_value"])
    sym = (Fore.GREEN + "✓" + Style.RESET_ALL) if p_all < 0.05 else (Fore.RED + "✗" + Style.RESET_ALL)
    print(f"\nSpearman correlation (trial order vs {acoustic_measure}):")
    print(f"  r = {r_all:.3f}, p = {p_all:.4f} {sym}")

    if not generate_figure:
        return

    group_trend = trial_means.groupby("trial_order")["mean_value"].agg(["mean", "sem"])
    m, b, _, _, _ = linregress(trial_means["trial_order"], trial_means["mean_value"])
    x_line = np.array([trial_means["trial_order"].min(), trial_means["trial_order"].max()])
    p_str = f"{p_all:.3f}" if p_all >= 0.001 else "< .001"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(group_trend.index,
                    group_trend["mean"] - group_trend["sem"],
                    group_trend["mean"] + group_trend["sem"],
                    alpha=0.2, color="#4c72b0")
    ax.plot(group_trend.index, group_trend["mean"],
            color="#4c72b0", linewidth=2, label="Mean ± SEM")
    ax.plot(x_line, b + m * x_line,
            color="red", linewidth=1.5, linestyle="--",
            label=f"Overall: r = {r_all:.3f}, p = {p_str}")
    ax.set_xlabel("Trial order")
    ax.set_ylabel(acoustic_measure.capitalize())
    ax.set_title(f"Group-level trial order trend — {acoustic_measure} (mean ± SEM across subjects)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_figure(fig, PATH_FIGURES,
                f"fig{str(n).zfill(2)}_{acoustic_measure}_by_trial_order.png", **SAVE_KW)
    plt.close(fig)


def acoustic_measure_by_trial_order_by_subject(n, acoustic_measure, filtered="unfiltered", generate_figure=True):
    print_section_title(f"{n}. {acoustic_measure.title()} by trial order — by subject")

    df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}", f"{SYSTEM.lower()}_velocity_{acoustic_measure}.gzip"))
    df = df[df["measure"] == acoustic_measure].copy()
    df = df[df["value"] >= 1e-5]
    if acoustic_measure == "pitch":
        df = df[df["value"] >= 75]

    trial_means = df.groupby(["subject", "trial"])["value"].mean().reset_index()
    trial_means = trial_means.rename(columns={"value": "mean_value"})
    trial_means["trial_order"] = trial_means.groupby("subject")["trial"].transform(
        lambda x: x.astype(str).map({v: i + 1 for i, v in enumerate(sorted(x.astype(str).unique()))})
    )

    subjects = sorted(df["subject"].unique())
    all_subjects = subjects.copy()
    colors = plt.cm.tab20.colors
    subject_colors = {s: colors[i % len(colors)] for i, s in enumerate(all_subjects)}

    print(f"\nSpearman correlation (trial order vs {acoustic_measure}) by subject:")
    for s in subjects:
        sub = trial_means[trial_means["subject"] == s]
        r, p = spearmanr(sub["trial_order"], sub["mean_value"])
        short = s.split("_", 1)[-1] if "_" in s else s
        if p < 0.05:
            sym = (Fore.BLUE + "↗" + Style.RESET_ALL) if r > 0 else (Fore.YELLOW + "↘" + Style.RESET_ALL)
        else:
            sym = Fore.RED + "✗" + Style.RESET_ALL
        print(f"  {short}: r = {r:.3f}, p = {p:.4f} {sym}")

    r_all, p_all = spearmanr(trial_means["trial_order"], trial_means["mean_value"])
    print(f"\nOverall: r = {r_all:.3f}, p = {p_all:.4f}")

    if not generate_figure:
        return

    n_subjects = len(subjects)
    n_cols = 6
    n_rows = math.ceil(n_subjects / n_cols)

    fig, axes = plt.subplots(4, 6, figsize=(20, 14), sharey=True)
    axes = axes.flatten()

    for i, s in enumerate(subjects):
        ax = axes[i]
        sub = trial_means[trial_means["subject"] == s].sort_values("trial_order")
        color = subject_colors[s]
        short = s.split("_", 1)[-1] if "_" in s else s

        ax.scatter(sub["trial_order"], sub["mean_value"],
                   color=color, alpha=0.5, s=25, zorder=3)

        m, b, _, _, _ = linregress(sub["trial_order"], sub["mean_value"])
        r_sp, p_sp = spearmanr(sub["trial_order"], sub["mean_value"])
        x_line = np.array([sub["trial_order"].min(), sub["trial_order"].max()])
        lw = 2.5 if p_sp < 0.05 else 1.5
        ls = "-" if p_sp < 0.05 else "--"
        ax.plot(x_line, m * x_line + b, color=color, lw=lw, linestyle=ls)

        sig = "p<.05" if p_sp < 0.05 else "ns"
        ax.set_title(f"{short}  (r = {r_sp:.2f}, {sig})", fontsize=11)
        ax.set_xlabel("Trial order")
        ax.set_ylabel(acoustic_measure.capitalize() if i % n_cols == 0 else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide unused axes
    for j in range(n_subjects, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Trial order vs {acoustic_measure} (solid = p < .05)", fontsize=13, y=1.01)
    plt.tight_layout()
    save_figure(fig, PATH_FIGURES,
                f"fig{str(n).zfill(2)}_{acoustic_measure}_by_trial_order_by_subject.png", **SAVE_KW)
    plt.close(fig)

def acoustic_measure_by_cartoon(n, acoustic_measure, filtered="unfiltered", generate_figure=True):
    print_section_title(f"{n}. {acoustic_measure.upper()} BY CARTOON")

    df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}", f"{SYSTEM.lower()}_velocity_{acoustic_measure}.gzip"))
    df = df[df["measure"] == acoustic_measure].copy()
    df = df[df["value"] >= 1e-5]
    if acoustic_measure == "pitch":
        df = df[df["value"] >= 75]

    # Load show mapping from info file
    df_info = pd.read_excel(PATH_INFO)
    cartoon_to_show = df_info.drop_duplicates("Retelling").set_index("Retelling")["Show"]

    # Per-trial means
    trial_means = (df.groupby(["subject", "trial"])["value"]
                   .mean().reset_index()
                   .rename(columns={"value": "mean_value", "trial": "cartoon"}))
    trial_means["show"] = trial_means["cartoon"].map(cartoon_to_show)

    by_cartoon = (trial_means.groupby("cartoon")["mean_value"]
                  .agg(["mean", "std", "min", "max", "count"]))
    by_cartoon.columns = ["Mean", "SD", "Min", "Max", "N trials"]
    by_cartoon = by_cartoon.round(4)

    print(f"Descriptive statistics per cartoon (N = {len(by_cartoon)} cartoons):")
    print(by_cartoon.to_string())

    print(f"\nSummary across cartoons:")
    print(f"  Lowest  mean : {by_cartoon['Mean'].min():.4f}  ({by_cartoon['Mean'].idxmin()})")
    print(f"  Highest mean : {by_cartoon['Mean'].max():.4f}  ({by_cartoon['Mean'].idxmax()})")
    print(f"  Grand mean   : {by_cartoon['Mean'].mean():.4f}  (SD = {by_cartoon['Mean'].std():.4f})")
    print(f"  Average N per cartoon : {by_cartoon['N trials'].mean():.1f}"
          f"  (range: {int(by_cartoon['N trials'].min())}–{int(by_cartoon['N trials'].max())})")

    # Kruskal-Wallis
    groups = [grp["mean_value"].values for _, grp in trial_means.groupby("cartoon")]
    k = len(groups)
    n_total = len(trial_means)
    h_stat, p_kruskal = kruskal(*groups)
    eta2 = h_stat / (n_total - 1)
    p_str = f"{p_kruskal:.4f}" if p_kruskal >= 0.001 else "< .001"

    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_kruskal < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"\nKruskal-Wallis: H({k - 1}) = {h_stat:.3f}, p = {p_str}  {sym}")
    print(f"Effect size: η² = {eta2:.3f}")

    # LMM likelihood ratio test
    print("\nLMM likelihood ratio test (controlling for subject):")
    lmm_null = mixedlm("mean_value ~ 1", trial_means,
                            groups=trial_means["subject"]).fit(reml=False)
    lmm_full = mixedlm("mean_value ~ C(cartoon)", trial_means,
                            groups=trial_means["subject"]).fit(reml=False)
    lr_stat = 2 * (lmm_full.llf - lmm_null.llf)
    from scipy.stats import chi2
    p_lr = chi2.sf(lr_stat, df=k - 1)
    p_lr_str = f"{p_lr:.4f}" if p_lr >= 0.001 else "< .001"
    sym_lr = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_lr < 0.05
              else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"  χ²({k - 1}) = {lr_stat:.3f}, p = {p_lr_str} {sym_lr}")

    # Post-hoc
    if p_kruskal < 0.05:
        import scikit_posthocs as sp
        dunn = sp.posthoc_dunn(trial_means, val_col="mean_value",
                               group_col="cartoon", p_adjust="holm")
        print("\nPost-hoc: Dunn's test (Holm correction)")
        print(dunn.to_string())
    else:
        print(f"\nNo significant difference in {acoustic_measure} across cartoons.")

    if not generate_figure:
        return

    cartoon_show = trial_means.groupby("cartoon")["show"].first()
    plot_df = (trial_means.groupby("cartoon")["mean_value"]
               .mean().to_frame().join(cartoon_show))
    plot_df = plot_df.sort_values("mean_value")
    cartoon_to_y = {c: i for i, c in enumerate(plot_df.index)}

    all_subjects = sorted(df["subject"].unique())
    colors = plt.cm.tab10.colors
    subject_colors = {s: colors[i % len(colors)] for i, s in enumerate(all_subjects)}

    fig, ax = plt.subplots(figsize=(10, max(8, len(plot_df) * 0.4)))
    y_pos = np.arange(len(plot_df))
    bar_colors = [SHOW_COLORS[s] for s in plot_df["show"]]

    ax.barh(y_pos, plot_df["mean_value"],
            color=bar_colors, edgecolor="white", height=0.75)

    # Per-subject per-cartoon means as dots
    for subj, grp in trial_means.groupby("subject"):
        subj_cartoon_means = grp.groupby("cartoon")["mean_value"].mean()
        ys = [cartoon_to_y[c] for c in subj_cartoon_means.index]
        ax.scatter(subj_cartoon_means.values, ys,
                   color=subject_colors[subj], s=18, zorder=3,
                   alpha=0.85, linewidths=0.3, edgecolors="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index, fontsize=7)
    ax.set_xlabel(acoustic_measure.capitalize())
    ax.set_title(
        f"{acoustic_measure.capitalize()} by cartoon (N = {len(plot_df)}, sorted by mean)\n"
        f"Kruskal-Wallis: H({k - 1}) = {h_stat:.2f}, p = {p_str}, η² = {eta2:.3f}"
    )

    from matplotlib.patches import Patch
    import matplotlib.lines as mlines
    show_handles = [Patch(facecolor=SHOW_COLORS[s], label=s) for s in SHOW_COLORS]
    subject_handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=subject_colors[s], markersize=6,
                      label=s.split("_", 1)[-1] if "_" in s else s)
        for s in all_subjects
    ]
    ax.legend(handles=show_handles + subject_handles,
              title="Show / Subject", fontsize=9, title_fontsize=9,
              loc="lower right")

    fig.tight_layout()
    save_figure(fig, PATH_FIGURES,
                f"fig{str(n).zfill(2)}_{acoustic_measure}_by_cartoon.png", **SAVE_KW)
    plt.close(fig)

def acoustic_measure_by_show(n, acoustic_measure, filtered="unfiltered", generate_figure=True):
    print_section_title(f"{n}. {acoustic_measure.upper()} BY SHOW")

    df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}", f"{SYSTEM.lower()}_velocity_{acoustic_measure}.gzip"))
    df = df[df["measure"] == acoustic_measure].copy()
    df = df[df["value"] >= 1e-5]
    if acoustic_measure == "pitch":
        df = df[df["value"] >= 75]

    # Load show mapping
    df_info = pd.read_excel(PATH_INFO)
    cartoon_to_show = df_info.drop_duplicates("Retelling").set_index("Retelling")["Show"]

    # Per-trial means
    trial_means = (df.groupby(["subject", "trial"])["value"]
                   .mean().reset_index()
                   .rename(columns={"value": "mean_value", "trial": "cartoon"}))
    trial_means["show"] = trial_means["cartoon"].map(cartoon_to_show)

    # Descriptive statistics
    by_show = (trial_means.groupby("show")["mean_value"]
               .agg(["mean", "std", "min", "max", "count"]))
    by_show.columns = ["Mean", "SD", "Min", "Max", "N trials"]
    by_show = by_show.round(4)

    print(f"Descriptive statistics per show (N = {len(by_show)} shows):")
    print(by_show.to_string())

    # Friedman test
    shows_ordered = sorted(trial_means["show"].unique())
    subj_show_means = (trial_means.groupby(["subject", "show"])["mean_value"]
                       .mean().unstack("show"))
    chi2_f, p_f = friedmanchisquare(*[subj_show_means[s].values
                                            for s in shows_ordered])
    p_str = f"{p_f:.4f}" if p_f >= 0.001 else "< .001"
    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_f < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"\nFriedman test: χ²({len(shows_ordered) - 1}) = {chi2_f:.3f}, "
          f"p = {p_str} {sym}")

    # Pairwise Wilcoxon post-hoc
    pval_lookup = {}
    if p_f < 0.05:
        print("\nPost-hoc: Pairwise Wilcoxon signed-rank tests (Holm correction)")
        pairs = list(combinations(shows_ordered, 2))
        pair_ps = []
        for s1, s2 in pairs:
            _, p = wilcoxon(subj_show_means[s1], subj_show_means[s2])
            pair_ps.append(p)

        reject, p_adj, _, _ = multipletests(pair_ps, method="holm")
        for (s1, s2), p_raw, p_adj_, sig in zip(pairs, pair_ps, p_adj, reject):
            pval_lookup[(s1, s2)] = p_adj_
            sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if sig
                   else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
            p_raw_str = f"{p_raw:.4f}" if p_raw >= 0.001 else "< .001"
            p_adj_str = f"{p_adj_:.4f}" if p_adj_ >= 0.001 else "< .001"
            print(f"  {s1} vs {s2}: p_raw = {p_raw_str}, p_adj = {p_adj_str} {sym}")
    else:
        print(f"\nNo significant difference in {acoustic_measure} across shows.")

    if not generate_figure:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=trial_means, x="show", y="mean_value",
                order=shows_ordered, hue="show", palette=SHOW_COLORS,
                width=0.5, fliersize=0, legend=False, ax=ax)
    sns.stripplot(data=trial_means, x="show", y="mean_value",
                  order=shows_ordered, hue="show", palette=SHOW_COLORS,
                  alpha=0.3, jitter=True, size=3, legend=False, ax=ax)

    # Significance brackets
    if pval_lookup:
        y_max = trial_means["mean_value"].max()
        y_range = y_max - trial_means["mean_value"].min()
        bracket_base = y_max + y_range * 0.05
        bracket_step = y_range * 0.08
        tick = y_range * 0.015

        def get_stars(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            return None

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

    sig_str = f"p = {p_str}" if p_f < 0.05 else "ns"
    ax.set_xlabel("Show")
    ax.set_ylabel(acoustic_measure.capitalize())
    ax.set_title(f"{acoustic_measure.capitalize()} by show ({sig_str})")

    fig.tight_layout()
    save_figure(fig, PATH_FIGURES,
                f"fig{str(n).zfill(2)}_{acoustic_measure}_by_show.png", **SAVE_KW)
    plt.close(fig)

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

def _load_acoustic_psd_dfs(measures, filtered="filtered", file_trail=""):
    """Loads one dataframe per acoustic measure into a dict {measure: df}."""
    dfs = {}
    for measure in measures:
        df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}",
                                     f"{SYSTEM.lower()}_velocity_{measure}{file_trail}.gzip"))
        df = df[df["measure"] == measure].copy()
        dfs[measure] = df
    return dfs


def _compute_acoustic_peak_stats(psds, freqs, freq_range=None, sd_line_height=0.8, ylim=None):
    """Per-measure peak frequency statistics: mean and SD across subjects' individual peaks,
    plus a y-position for the horizontal SD line at `sd_line_height` (fraction of that measure's
    OWN y-axis range, 0-1) — each measure gets its own height, since envelope/intensity/pitch/F1/F2
    are on different units and don't share a y-scale (unlike kinematic joints, which all share
    velocity units and one plot_body_graphs scale)."""
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    else:
        mask = np.ones_like(freqs, dtype=bool)
    masked_freqs = freqs[mask]

    peak_stats = {}
    for measure, subj_psds in psds.items():
        if not subj_psds:
            continue

        if ylim is not None:
            y_min, y_max = ylim
        else:
            means_masked = [mean[mask] for mean, _ in subj_psds.values()]
            y_min = float(np.nanmin([np.nanmin(m) for m in means_masked]))
            y_max = float(np.nanmax([np.nanmax(m) for m in means_masked]))
        y_fixed = y_min + sd_line_height * (y_max - y_min)

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

        peak_stats[measure] = {"mean_freq": mean_freq, "sd_freq": sd_freq, "y_at_peak": y_fixed}
    return peak_stats


def _infer_acoustic_fs(dfs):
    """Same logic as kinematics' _infer_fs, but works off the first available measure's dataframe."""
    df = next(iter(dfs.values()))
    subj = df["subject"].iloc[0]
    trial = df["trial"].iloc[0]
    group = df[(df["subject"] == subj) & (df["trial"] == trial)]
    timestamps = group.sort_values("timestamp")["timestamp"].values
    return 1 / np.median(np.diff(timestamps))


def _filter_acoustic_dfs(dfs, subjects, trials, exclude_subjects=None):
    filtered = {}
    for measure, df in dfs.items():
        if subjects is not None:
            subj_list = [subjects] if isinstance(subjects, str) else subjects
            df = df[df["subject"].isin(subj_list)]
        if exclude_subjects is not None:
            exclude_list = [exclude_subjects] if isinstance(exclude_subjects, str) else exclude_subjects
            df = df[~df["subject"].isin(exclude_list)]
        if trials is not None:
            trial_list = [trials] if isinstance(trials, str) else trials
            df = df[df["trial"].isin(trial_list)]
        filtered[measure] = df
    n_subj = len(set().union(*[set(df["subject"].unique()) for df in filtered.values()]))
    print(f"\t{n_subj} subject(s) across {len(filtered)} measure(s)")
    return filtered


def _get_acoustic_psds(dfs, fs, nperseg, normalization, band_max_range, psd_unit, trial_average="trial"):
    """Returns {measure: {subject: (mean_psd, sd_psd)}}, freqs.

    trial_average:
        "trial"  — compute one Welch PSD per trial, then average across trials (default;
                   avoids concatenation artifacts at trial boundaries).
        "concat" — concatenate all trials for a subject into one signal, then compute a
                   single Welch PSD on the concatenated series (matches the original
                   plot_acoustic_psd behavior). No per-trial SD is available in this mode.
    """
    if trial_average not in ("trial", "concat"):
        raise ValueError(f"trial_average must be 'trial' or 'concat', got {trial_average!r}")

    measure_list = sorted(dfs.keys())
    psds = {}
    freqs = None

    for measure in measure_list:
        df = dfs[measure]
        subj_list = sorted(df["subject"].unique())
        psds[measure] = {}

        for subj in subj_list:
            rows = df[df["subject"] == subj]
            if rows.empty:
                continue

            if trial_average == "concat":
                values = rows.sort_values(["trial", "timestamp"])["value"].to_numpy()
                if len(values) < nperseg:
                    continue
                freqs, psd = welch(values, fs=fs, nperseg=nperseg)
                mean_psd = psd
                sd_psd = None  # no per-trial variance to report in concat mode
            else:
                trial_psds = []
                for trial in sorted(rows["trial"].unique()):
                    trial_values = rows[rows["trial"] == trial].sort_values("timestamp")["value"].to_numpy()
                    if len(trial_values) < nperseg:
                        continue
                    freqs, psd = welch(trial_values, fs=fs, nperseg=nperseg)
                    trial_psds.append(psd)
                if not trial_psds:
                    continue
                mean_psd = np.mean(trial_psds, axis=0)
                sd_psd = np.std(trial_psds, axis=0) if len(trial_psds) > 1 else None

            psds[measure][subj] = (mean_psd, sd_psd)

        if not psds[measure]:
            continue

        global_max = np.max([psds[measure][s][0].max() for s in psds[measure]])
        subjects_max = {s: psds[measure][s][0].max() for s in psds[measure]}

        for subj in psds[measure]:
            mean_psd, sd_psd = psds[measure][subj]
            psds[measure][subj] = _normalize(mean_psd, sd_psd, freqs, normalization,
                                             global_max, subjects_max[subj], band_max_range)
            if psd_unit == "dB":
                mean_psd, sd_psd = psds[measure][subj]
                psds[measure][subj] = (10 * np.log10(mean_psd + 1e-30),
                                       10 * np.log10(sd_psd + 1e-30) if sd_psd is not None else None)

    return psds, freqs

def _stats_acoustic_subjects(psds, freqs, z_threshold=2.0, normalization=None):
    """Per-measure inter-subject diagnostics: CV of total power, z-score outlier flags, Spearman
    inter-subject PSD similarity, and (for peak-normalized data only) decay at 1 Hz relative to peak —
    each computed WITHIN a measure, never pooled across measures."""
    print_section_title("STATS — ACOUSTIC SUBJECT VARIABILITY")

    idx_1hz = np.argmin(np.abs(freqs - 1.0))
    show_decay = normalization in ("subject_peak", "global_peak")

    for measure, subj_psds in psds.items():
        subj_list = sorted(subj_psds.keys())
        if len(subj_list) < 2:
            continue

        band_power = {s: np.trapezoid(subj_psds[s][0], freqs) for s in subj_list}
        values = np.array(list(band_power.values()))
        cv = values.std() / values.mean() if values.mean() > 0 else np.nan
        print(f"\n  {measure}: CV (SD/mean of total power) = {cv:.3f}" if not np.isnan(cv)
              else f"\n  {measure}: CV = nan")

        psd_matrix = np.array([subj_psds[s][0] for s in subj_list])
        mean_across = psd_matrix.mean(axis=0)
        sd_across = psd_matrix.std(axis=0)
        z_matrix = np.where(sd_across > 0, (psd_matrix - mean_across) / sd_across, 0)
        mean_z_per_subj = z_matrix.mean(axis=1)
        for i, subj in enumerate(subj_list):
            if abs(mean_z_per_subj[i]) > z_threshold:
                direction = "↑" if mean_z_per_subj[i] > 0 else "↓"
                print(f"    {direction} {subj}: mean z = {mean_z_per_subj[i]:.2f}")

        if len(subj_list) >= 2:
            from itertools import combinations
            pairs = list(combinations(range(len(subj_list)), 2))
            z_vals = []
            for i, j in pairs:
                r, _ = spearmanr(psd_matrix[i], psd_matrix[j])
                z_vals.append(np.arctanh(np.clip(r, -0.9999, 0.9999)))
            r_avg = np.tanh(np.mean(z_vals))
            _, p = ttest_1samp(z_vals, 0)
            p_str = "< .001" if p < 0.001 else f"{p:.4f}"
            print(f"    Inter-subject similarity: n pairs = {len(pairs)}, mean r = {r_avg:.3f}, p = {p_str}")

        if show_decay:
            pct_decayed = [(1 - psd_matrix[i, idx_1hz]) * 100 for i in range(len(subj_list))]
            print(f"    Decay at 1 Hz (relative to peak):")
            for subj, pct in zip(subj_list, pct_decayed):
                print(f"      {subj}: {pct:.1f}%")
            print(f"      Mean: {np.mean(pct_decayed):.1f}%, SD: {np.std(pct_decayed):.1f}%")


def _get_acoustic_plot_dictionary(psds, freqs, show_average, freq_log, show_sd):
    if freq_log:
        mask = freqs > 0
        freqs = freqs[mask]
        psds = {measure: {subj: (mean[mask], sd[mask] if sd is not None else None)
                          for subj, (mean, sd) in subj_psds.items()}
                for measure, subj_psds in psds.items()}

    plot_dictionary = {}
    for measure, subj_psds in psds.items():
        graph = Graph()

        if show_average == "bottom" and subj_psds:
            means = [mean for mean, _ in subj_psds.values()]
            mean_psd = np.nanmean(means, axis=0)
            sd_psd = np.nanstd(means, axis=0) if show_sd and len(means) > 1 else None
            graph.add_plot(freqs, mean_psd, sd_psd, 2.5, "-", "black", "Mean")

        for subj, (mean_psd, sd_psd) in subj_psds.items():
            sd = sd_psd if show_sd else None
            short = subj.split("_", 1)[-1] if "_" in subj else subj
            graph.add_plot(freqs, mean_psd, sd, 0.8, "-", SUBJECT_COLORS[subj], short)

        if show_average == "top" and subj_psds:
            means = [mean for mean, _ in subj_psds.values()]
            mean_psd = np.nanmean(means, axis=0)
            sd_psd = np.nanstd(means, axis=0) if show_sd and len(means) > 1 else None
            graph.add_plot(freqs, mean_psd, sd_psd, 2.5, "-", "black", "Mean")

        plot_dictionary[measure] = graph

    return plot_dictionary

def _plot_acoustic_measure_on_ax(ax, graph, freq_min, freq_max, x_scale_log=False, show_legend=True,
                                 title=None, peak_stats=None, peak_line_color="#202020",
                                 peak_line_width=1.2, peak_line_style="--", sd_color=None, sd_width=1.5, ylim=None):
    for subplot in graph.plots:
        is_mean = subplot.label == "Mean"
        line, = ax.plot(subplot.x, subplot.y, linewidth=subplot.line_width,
                       linestyle=subplot.line_style, color=subplot.color, label=subplot.label,
                       alpha=1.0 if is_mean else 0.7, zorder=5 if is_mean else 2)
        if subplot.sd is not None:
            ax.fill_between(subplot.x, subplot.y - subplot.sd, subplot.y + subplot.sd,
                           alpha=0.3, facecolor=line.get_color())

    ax.set_xlim(freq_min, freq_max)
    ax.set_xticks(np.arange(0, freq_max + 1, 1))
    if x_scale_log:
        ax.set_xscale("log")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Normalized amplitude", fontsize=11)
    if ylim is not None:
        ax.set_ylim(ylim)
    print(f"1. YLIM: {ylim}")
    print(f"2. YLIM: {ax.get_ylim()}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(fontsize=8, ncol=2, loc="upper right")

    # ── Y-axis limits based on visible x-range only ──────────────────────────
    y_visible = []
    for subplot in graph.plots:
        mask = (subplot.x >= freq_min) & (subplot.x <= freq_max)
        if mask.any():
            y_visible.append(subplot.y[mask])
    if y_visible and ylim is None:
        y_all = np.concatenate(y_visible)
        y_min, y_max = y_all.min(), y_all.max()
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # ── Peak frequency marker (vertical line + horizontal SD bar) ────────────
    if peak_stats is not None:
        if sd_color is None:
            sd_color = peak_line_color
        freq, sd = peak_stats["mean_freq"], peak_stats["sd_freq"]
        y = peak_stats["y_at_peak"]
        ax.axvline(freq, color=peak_line_color, lw=peak_line_width, ls=peak_line_style, zorder=6)
        ax.plot([freq - sd, freq + sd], [y, y], color=sd_color, lw=sd_width, zorder=6)
    print(f"3. YLIM: {ax.get_ylim()}")


def plot_acoustic_multi_panel(plot_dictionary, freqs, measure_order=None, ncols=None,
                              figsize=None, freq_min=0, freq_max=10.0, x_scale_log=False,
                              show_legend=True, path_save=None, show=True, peak_stats_dict=None,
                              peak_line_color="#202020", peak_line_width=1.2, peak_line_style="--",
                              ylim=None):
    measure_order = measure_order or sorted(plot_dictionary.keys())
    n = len(measure_order)
    ncols = ncols or n
    nrows = math.ceil(n / ncols)
    figsize = figsize or (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, measure in enumerate(measure_order):
        ax = axes_flat[idx]
        peak_stats = peak_stats_dict.get(measure) if peak_stats_dict else None
        _plot_acoustic_measure_on_ax(ax, plot_dictionary[measure], freq_min, freq_max,
                                     x_scale_log, show_legend, measure.capitalize(),
                                     peak_stats, peak_line_color, peak_line_width, peak_line_style, ylim)

    for idx in range(n, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout()
    if path_save is not None:
        folder = op.split(path_save)[0]
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        plt.savefig(path_save)
    if show:
        plt.show()
    plt.close(fig)

def plot_acoustic_psd(n, measures=None, fs=None, filtered="filtered", file_trail="",
                      subjects=None, trials=None, exclude_subjects=None, nperseg=256, normalization=None,
                      band_max_range=(0.5, 4.0), psd_unit="default", freq_log=False,
                      freq_min=0, freq_max=10.0, show_stats=True, show_average=True, show_sd=False,
                      trial_average="trial", ncols=None, show_legend=True, show=True, path_png=None, ax=None,
                      mark_peak_freq=False, peak_freq_range=None, sd_line_height=0.8, ylim=None,
                      peak_line_color="#202020", peak_line_width=1.2, peak_line_style="--"):

    print_section_title(f"{n}. PSD — ACOUSTIC MEASURES")
    if isinstance(measures, str):
        measures = [measures]
    measures = measures or DERIVATIVES

    if ax is not None and len(measures) != 1:
        raise ValueError("When drawing onto an external `ax`, `measures` must contain exactly one "
                        "measure (one axis = one measure) — for multiple measures, omit `ax` and "
                        "let the function build its own multi-panel figure.")

    dfs = _load_acoustic_psd_dfs(measures, filtered, file_trail)
    dfs = _filter_acoustic_dfs(dfs, subjects, trials, exclude_subjects)
    if fs is None:
        fs = _infer_acoustic_fs(dfs)

    psds, freqs = _get_acoustic_psds(dfs, fs, nperseg, normalization, band_max_range, psd_unit, trial_average)

    if show_stats:
        _stats_acoustic_subjects(psds, freqs, normalization=normalization)

    peak_stats_dict = None
    if mark_peak_freq:
        peak_stats_dict = _compute_acoustic_peak_stats(psds, freqs, peak_freq_range, sd_line_height, ylim)
        print("\n  Peak frequency per measure (mean ± SD across subjects):")
        for measure in measures:
            stats = peak_stats_dict.get(measure)
            if stats is None:
                continue
            print(f"    {measure:<10} {stats['mean_freq']:.2f} ± {stats['sd_freq']:.2f} Hz")

    plot_dictionary = _get_acoustic_plot_dictionary(psds, freqs, show_average, freq_log, show_sd)

    if ax is not None:
        measure = measures[0]
        peak_stats = peak_stats_dict.get(measure) if peak_stats_dict else None
        _plot_acoustic_measure_on_ax(ax, plot_dictionary[measure], freq_min, freq_max,
                                     freq_log, show_legend, measure.capitalize(),
                                     peak_stats, peak_line_color, peak_line_width, peak_line_style, ylim=ylim)
        return

    plot_acoustic_multi_panel(plot_dictionary, freqs, measure_order=measures, ncols=ncols,
                              freq_min=freq_min, freq_max=freq_max, x_scale_log=freq_log,
                              show_legend=show_legend, path_save=path_png, show=show,
                              peak_stats_dict=peak_stats_dict, peak_line_color=peak_line_color,
                              peak_line_width=peak_line_width, peak_line_style=peak_line_style, ylim=ylim)

def plot_acoustic_psd_old(n, acoustic_measure, fs=20, nperseg=256, freq_min=0, freq_max=10.0,
                      freq_log=False, psd_unit="default", trials=None, show_average=True, show_legend=True,
                      subjects_filter=None, normalization=None, band_max_range=(0.5, 4.0),
                      filtered="filtered", generate_figure=True,
                      file_trail="", ax=None, show=True):

    print_section_title(f"{n}. PSD — {acoustic_measure.upper()}")

    df = pd.read_parquet(op.join(PATH_DATAFRAME, f"FS{int(FS)} {filtered}", f"{SYSTEM.lower()}_velocity_{acoustic_measure}.gzip"))
    df = df[df["measure"] == acoustic_measure].copy()
    # df = df[df["value"] >= 1e-5]
    # if acoustic_measure == "pitch":
    #     df = df[df["value"] >= 75]

    if trials is not None:
        trials = [trials] if isinstance(trials, str) else list(trials)
        df = df[df["trial"].isin(trials)]
    if subjects_filter is not None:
        subjects_filter = [subjects_filter] if isinstance(subjects_filter, str) else list(subjects_filter)
        df = df[df["subject"].isin(subjects_filter)]

    subj_list = sorted(df["subject"].unique())
    print(f"  {len(subj_list)} subject(s), {len(df)} rows\n")

    # ── Frequency axis ────────────────────────────────────────────────────────
    freqs_full, _ = welch(np.zeros(nperseg), fs=fs, nperseg=nperseg)
    mask = freqs_full <= freq_max
    if freq_log:
        mask = mask & (freqs_full > 0)
    freqs_plot = freqs_full[mask]

    # ── Colors ────────────────────────────────────────────────────────────────
    cmap = plt.get_cmap("tab20", max(len(subj_list), 1))

    # ── Raw PSD per subject (unnormalized at this stage) ────────────────────────
    external_ax = ax is not None
    if not external_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    all_psds = []
    subj_labels = []
    for subj in subj_list:
        rows = df[df["subject"] == subj]
        if rows.empty:
            continue

        values = rows.sort_values(["trial", "timestamp"])["value"].to_numpy()
        if len(values) < nperseg:
            print(f"  {subj}: only {len(values)} samples, skipping")
            continue

        _, psd = welch(values, fs=fs, nperseg=nperseg)
        psd = psd[mask]
        if psd_unit == "dB":
            psd = 10 * np.log10(psd + 1e-30)

        all_psds.append(psd)
        subj_labels.append(subj)

    if not all_psds:
        print("  No valid data found.")
        plt.close(fig)
        return

    all_psds_arr = np.array(all_psds)
    mean_psd = np.nanmean(all_psds_arr, axis=0)
    sd_psd = np.nanstd(all_psds_arr, axis=0)

    # ── Normalization ────────────────────────────────────────────────────────
    # subject_max: average of each subject's own peak PSD value
    # global_max: peak of the group-averaged PSD curve
    subject_max = np.mean([psd.max() for psd in all_psds])
    global_max = mean_psd.max()

    # Self-normalize each subject curve so lines remain comparable to the group mean
    normalized_psds = []
    for psd in all_psds:
        this_subject_max = psd.max()  # this subject's own peak, not the group average
        psd_norm, _ = _normalize(psd, None, freqs_plot, normalization, global_max, this_subject_max, band_max_range)
        normalized_psds.append(psd_norm)

    mean_psd, sd_psd = _normalize(mean_psd, sd_psd, freqs_plot, normalization, global_max, subject_max, band_max_range)

    for subj, psd_norm in zip(subj_labels, normalized_psds):
        short = subj.split("_", 1)[-1] if "_" in subj else subj
        ax.plot(freqs_plot, psd_norm, lw=0.8, alpha=0.7, color=SUBJECT_COLORS[subj], label=short)

    idx_1hz = np.argmin(np.abs(freqs_plot - 1.0))
    pct_at_1hz = [(1 - psd[idx_1hz]) * 100 for psd in normalized_psds]
    print(f"\n  Normalized power remaining at 1 Hz:")
    for subj, pct in zip(subj_labels, pct_at_1hz):
        short = subj.split("_", 1)[-1] if "_" in subj else subj
        print(f"    {short}: {pct:.1f}%")
    print(f"  Mean: {np.mean(pct_at_1hz):.1f}%, SD: {np.std(pct_at_1hz):.1f}%")

    # ── Inter-subject PSD similarity ──────────────────────────────────────────
    if len(normalized_psds) >= 2:
        from itertools import combinations
        pairs = list(combinations(range(len(normalized_psds)), 2))
        z_vals = []
        for i, j in pairs:
            r, _ = spearmanr(normalized_psds[i], normalized_psds[j])
            z_vals.append(np.arctanh(np.clip(r, -0.9999, 0.9999)))

        mean_z = np.mean(z_vals)
        r_avg = np.tanh(mean_z)
        _, p = ttest_1samp(z_vals, 0)
        p_str = "< .001" if p < 0.001 else f"{p:.4f}"

        print(f"\n  Inter-subject PSD similarity (Spearman, Fisher-averaged across pairs):")
        print(f"  n pairs = {len(pairs)}, mean r = {r_avg:.3f}, p = {p_str}")

    if show_average:
        ax.plot(freqs_plot, mean_psd, lw=2.5, color="black", label="Mean", zorder=5)

    idx_05 = np.argmin(np.abs(freqs_plot - 0.5))
    print(f"  n subjects    : {len(normalized_psds)}")
    print(f"  PSD at 0.5 Hz : {mean_psd[idx_05]:.4f}")
    print(f"  PSD mean      : {mean_psd.mean():.4f}")

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Normalized amplitude", fontsize=11)
    ax.set_xticks(np.arange(0, freq_max + 1, 1))
    ax.set_xlim(freq_min, freq_max)
    ax.set_title(f"{acoustic_measure.capitalize()}", fontsize=12, fontweight="bold")
    if freq_log:
        ax.set_xscale("log")
    if show_legend:
        ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Y-axis limits based on visible x-range only ──────────────────────────
    visible_mask = (freqs_plot >= freq_min) & (freqs_plot <= freq_max)
    all_visible_curves = normalized_psds + ([mean_psd] if show_average else [])
    y_visible = np.concatenate([psd[visible_mask] for psd in all_visible_curves])
    y_min, y_max = y_visible.min(), y_visible.max()
    y_pad = (y_max - y_min) * 0.05  # 5% padding
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    if not external_ax:
        plt.tight_layout()
        if show:
            plt.show()
        if generate_figure:
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_psd_{acoustic_measure}.png", **SAVE_KW)
        plt.close(fig)


if __name__ == "__main__":
    generate_figures = True

    # start = 1
    # for i in range(5):
    #     plot_acoustic_measure_per_participant(i + start, DERIVATIVES[i], generate_figure=generate_figures)
    #
    # voiced_unvoiced_ratio(6, generate_figure=generate_figures)
    acoustic_measure_correlations(7, generate_figure=generate_figures)

    # start = 8
    # for i in range(5):
    #     acoustic_measure_by_trial_order(i + start, DERIVATIVES[i], generate_figure=generate_figures)
    #
    # start = 13
    # for i in range(5):
    #     acoustic_measure_by_trial_order_by_subject(i + start, DERIVATIVES[i], generate_figure=generate_figures)
    #
    # start = 18
    # for i in range(5):
    #     acoustic_measure_by_cartoon(i + start, DERIVATIVES[i], generate_figure=generate_figures)
    #
    # start = 23
    # for i in range(5):
    #     acoustic_measure_by_show(i + start, DERIVATIVES[i], generate_figure=generate_figures)
    #
    # start = 28
    # for i in range(5):
    #     plot_acoustic_psd(i + start, DERIVATIVES[i], show=False, generate_figure=generate_figures)