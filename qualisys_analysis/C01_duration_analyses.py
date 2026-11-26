import math

import pandas as pd
from scipy import stats
import pingouin as pg
from colorama import Fore, Style
from constants import *
import os

import numpy as np
import seaborn as sns

from helper_functions import save_figure, print_section_title, plt
import statsmodels.formula.api as smf

PATH_FIGURES = op.join(PATH_PLOTS, "Duration")
os.makedirs(PATH_FIGURES, exist_ok=True)

SHOW_COLORS = {
    "Wile E. Coyote": "#ffab66",
    "Pink Panther": "#ff66ff",
    "Tom & Jerry": "#7ec6e7",
}
MASK_COLORS = {"Masked": "#4C72B0", "Unmasked": "#DD8452"}

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="whitegrid", font_scale=1.2, rc=custom_params)
SAVE_KW = dict(dpi=150, bbox_inches="tight")

from matplotlib import font_manager
from pathlib import Path
for path in Path(r"Metropolis").glob("*.otf"):
    font_manager.fontManager.addfont(str(path))

plt.rcParams["font.family"] = "Metropolis"


def test_duration_by_group(df_clean, group_col):
    by_group = df_clean.groupby(group_col)["Retelling duration"].agg(["mean", "std", "min", "max", "count"])
    by_group.columns = ["Mean (s)", "SD (s)", "Min (s)", "Max (s)", "N trials"]
    by_group = by_group.round(2)
    print(by_group.to_string())

    groups = [group["Retelling duration"].values for _, group in df_clean.groupby(group_col)]

    # Step 1: Normality (Shapiro-Wilk)
    print(f"\nNormality tests (Shapiro-Wilk) by {group_col.lower()}:")
    normality_results = {}
    for name, group in df_clean.groupby(group_col):
        stat, p = stats.shapiro(group["Retelling duration"])
        normality_results[name] = p
        print(
            f"  {name}: W = {stat:.3f}, p = {p:.4f} {Fore.GREEN + '\033[1m✓\033[0m' if p > 0.05 else Fore.RED + '\033[1m✗\033[0m'}",
            end="")
        print(Style.RESET_ALL)
    all_normal = all(p > 0.05 for p in normality_results.values())
    if all_normal:
        print(f"For all {group_col.lower()}s, the retelling durations follow a normal distribution.")
    else:
        print(f"For at least one {group_col.lower()}, the retelling durations do not follow a normal distribution.")

    # Step 2: Equal variances (Levene)
    if all_normal:
        lev_stat, lev_p = stats.levene(*groups)
        print(f"\nLevene's test (equal variances): W = {lev_stat:.3f}, p = {lev_p:.4f}")
        equal_var = lev_p > 0.05
        print("The variance in retelling duration is " +
              ("similar" if equal_var else "not similar") +
              f" across all {group_col.lower()}s.")
    else:
        equal_var = False

    # Step 3: Main test
    if all_normal and equal_var:
        print("\nAs the retelling durations " + Fore.GREEN + "follow a normal distribution " + Style.RESET_ALL +
              "and " + Fore.GREEN + "have similar variances " + Style.RESET_ALL +
              f"across {group_col.lower()}s, we perform " + "\033[1m" + "a one-way ANOVA." + "\033[0m\n")
        f_stat, p_main = stats.f_oneway(*groups)
        print(f"One-way ANOVA: F = {f_stat:.3f}, p = {p_main:.4f}")
        if p_main < 0.05:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            print("\nPost-hoc: Tukey HSD")
            tukey = pairwise_tukeyhsd(df_clean["Retelling duration"], df_clean[group_col], alpha=0.05)
            print(tukey)

    elif all_normal and not equal_var:
        print("\nAs the retelling durations " + Fore.GREEN + "follow a normal distribution " + Style.RESET_ALL +
              "but " + Fore.RED + "do not have similar variances " + Style.RESET_ALL +
              f"across {group_col.lower()}s, we perform " + "\033[1m" + "a Welch's ANOVA." + "\033[0m\n")
        welch = pg.welch_anova(data=df_clean, dv="Retelling duration", between=group_col)
        p_main = welch["p_unc"].values[0]
        print("Welch's ANOVA:")
        print(welch.to_string())
        if p_main < 0.05:
            print("\nPost-hoc: Games-Howell")
            gh = pg.pairwise_gameshowell(data=df_clean, dv="Retelling duration", between=group_col)
            print(gh.to_string())

    else:
        print("\nAs the retelling durations " + Fore.RED + "do not follow a normal distribution " + Style.RESET_ALL +
              "and " + Fore.RED + "do not have similar variances " + Style.RESET_ALL +
              f"across {group_col.lower()}s, we perform " + "\033[1m" + "a Kruskal-Wallis test." + "\033[0m\n")
        h_stat, p_main = stats.kruskal(*groups)
        print(f"Kruskal-Wallis: H = {h_stat:.3f}, p = {p_main:.4f}")
        if p_main < 0.05:
            print("\nPost-hoc: Dunn's test")
            dunn = pg.pairwise_tests(data=df_clean, dv="Retelling duration", between=group_col,
                                     parametric=False, padjust="holm")
            print(dunn.to_string())

    # Step 4: Kruskal-Wallis confirmation
    if all_normal:
        h_stat, p_kruskal = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis (non-parametric confirmation): H = {h_stat:.3f}, p = {p_kruskal:.4f}")


def overall_retelling_duration(n, df_clean, generate_figure=True):
    print_section_title(f"{n}. OVERALL RETELLING DURATION")

    print("Retelling duration across all trials")
    desc = df_clean["Retelling duration"].agg(["mean", "std", "min", "max"])
    print(f"  Mean : {desc['mean']:.2f} s")
    print(f"  SD   : {desc['std']:.2f} s")
    print(f"  Min  : {desc['min']:.2f} s")
    print(f"  Max  : {desc['max']:.2f} s")

    print("\nCartoon duration across all trials")
    desc = df_clean["Cartoon duration"].agg(["mean", "std", "min", "max"])
    print(f"  Mean : {desc['mean']:.2f} s")
    print(f"  SD   : {desc['std']:.2f} s")
    print(f"  Min  : {desc['min']:.2f} s")
    print(f"  Max  : {desc['max']:.2f} s")

    print("\nDifference between retelling and cartoon durations")
    diff = (df_clean["Retelling duration"] - df_clean["Cartoon duration"]).agg(["mean", "std", "min", "max"])
    print(f"  Mean : {diff['mean']:.2f} s")
    print(f"  SD   : {diff['std']:.2f} s")
    print(f"  Min  : {diff['min']:.2f} s")
    print(f"  Max  : {diff['max']:.2f} s")

    if generate_figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = df_clean["Retelling duration"]
        ax.hist(data, bins=20, color="#4C72B0", alpha=0.6, edgecolor="white", density=True, label="Trials")
        kde_x = np.linspace(data.min() - 5, data.max() + 5, 300)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color="#4C72B0", lw=2.5, label="KDE")
        ax.axvline(data.mean(), color="black", lw=2, linestyle="--", label=f"Mean = {data.mean():.1f} s")
        ax.axvline(data.mean() - data.std(), color="gray", lw=1.5, linestyle=":", label=f"±1 SD")
        ax.axvline(data.mean() + data.std(), color="gray", lw=1.5, linestyle=":")
        ax.set_xlabel("Retelling duration (s)")
        ax.set_ylabel("Density")
        ax.set_title("Overall retelling duration (N = 324 trials)")
        ax.legend()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_overall_duration.png", **SAVE_KW)

from matplotlib.colors import to_rgb
import numpy as np

def darken(hex_color, factor=0.65):
    rgb = np.array(to_rgb(hex_color))
    return tuple(rgb * factor)

def lighten(hex_color, factor=0.65):
    rgb = np.array(to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * factor)

def retelling_duration_by_subject(n, df_clean, generate_figure=True, ax=None):
    print_section_title(f"{n}. RETELLING DURATION BY SUBJECT")
    test_duration_by_group(df_clean, "Subject")

    # Define order and anon labels here so they're available for both stats and figure
    order = df_clean.groupby("Subject")["Retelling duration"].mean().sort_values().index.tolist()
    anon_labels = {s: f"Q{str(i+1).zfill(2)}" for i, s in enumerate(order)}

    # Identify non-normal subjects and their outlying trials
    print("\nOutlier detection (Shapiro-Wilk + IQR):")
    for subject, group in df_clean.groupby("Subject"):
        stat, p = stats.shapiro(group["Retelling duration"])
        if p < 0.05:
            q1, q3 = group["Retelling duration"].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = group[group["Retelling duration"] > q3 + 1.5 * iqr]
            print(f"  {anon_labels[subject]}: W = {stat:.3f}, p = {p:.4f} — {len(outliers)} outlier(s)")
            print(outliers[["Cartoon", "Retelling duration"]].to_string())

    if generate_figure:
        order = df_clean.groupby("Subject")["Retelling duration"].mean().sort_values().index.tolist()
        anon_labels = {s: f"Q{str(i+1).zfill(2)}" for i, s in enumerate(order)}

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
            standalone = True
        else:
            standalone = False

        bp = sns.boxplot(data=df_clean, x="Subject", y="Retelling duration",
                         order=order, width=0.5, fliersize=0, legend=False, ax=ax,
                         color="white")  # plain color, we recolor manually below

        for i, subject in enumerate(order):
            color = SUBJECT_COLORS[subject]
            dark = darken(color)
            # box patch
            ax.patches[i].set_facecolor(color)
            ax.patches[i].set_edgecolor(dark)
            ax.patches[i].set_linewidth(1.5)
            # median line (6 lines per box: whisker, whisker, cap, cap, median, flier)
            bp.lines[i * 6 + 4].set_color(dark)
            bp.lines[i * 6 + 4].set_linewidth(2)

        # Strip plot with matching colors
        for i, subject in enumerate(order):
            subj_data = df_clean[df_clean["Subject"] == subject]
            color = SUBJECT_COLORS[subject]
            ax.scatter(
                np.random.uniform(-0.2, 0.2, len(subj_data)) + i,
                subj_data["Retelling duration"],
                color=lighten(color, factor=0.8), edgecolors=darken(color, factor=0.8),
                linewidths=0.4, alpha=0.6, s=12, zorder=3
            )

        # Color boxes: fill from SUBJECT_COLORS, edge and median darker
        for patch, subject in zip(ax.patches[:len(order)], order):
            color = SUBJECT_COLORS[subject]
            patch.set_facecolor(color)
            patch.set_edgecolor(darken(color))
            patch.set_linewidth(1.5)

        # Color medians darker
        for line, subject in zip(bp.lines[4::6], order):  # median is every 6th line starting at index 4
            line.set_color(darken(SUBJECT_COLORS[subject]))
            line.set_linewidth(2)

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([anon_labels[s] for s in order], rotation=0, fontsize=10)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Retelling duration (s)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        if standalone:
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_duration_by_subject.png", **SAVE_KW)
            plt.close(fig)


def trial_order(n, df_clean, generate_figure=True, ax=None):
    print_section_title(f"{n}. TRIAL ORDER EFFECT")

    # Pearson is preferred here: session order is a metric integer variable and the
    # relationship with duration is expected to be linear.
    r_all, p_all = stats.pearsonr(df_clean["Session order"], df_clean["Retelling duration"])
    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_all < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"\nPearson correlation (session order vs retelling duration):")
    print(f"  r = {r_all:.3f}, p = {p_all:.4f} {sym}")

    print("\nLinear Mixed Model: Retelling duration ~ Session order + (1|Subject)")
    lmm = smf.mixedlm("Q('Retelling duration') ~ Q('Session order')",
                       df_clean, groups=df_clean["Subject"])
    lmm_fit = lmm.fit(reml=True)
    lmm_beta = lmm_fit.params["Q('Session order')"]
    lmm_p    = lmm_fit.pvalues["Q('Session order')"]
    lmm_t = lmm_fit.tvalues["Q('Session order')"]
    lmm_se = lmm_fit.bse["Q('Session order')"]
    lmm_ci   = lmm_fit.conf_int().loc["Q('Session order')"]
    sym = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if lmm_p < 0.05
           else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"  β(Session order) = {lmm_beta:.4f}, "
          f"95% CI [{lmm_ci[0]:.4f}, {lmm_ci[1]:.4f}], "
          f"p = {lmm_p:.4f} {sym}")
    print(f"  t = {lmm_t:.3f}, SE = {lmm_se:.4f}")
    direction = "increase" if lmm_beta > 0 else "decrease"
    print(f"  Interpretation: retelling duration tends to {direction} by "
          f"{abs(lmm_beta):.4f} s per trial "
          f"({'significant' if lmm_p < 0.05 else 'non-significant'})")

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

        group_trend = (df_clean.groupby("Session order")["Retelling duration"]
                       .agg(["mean", "sem"]))
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            standalone = True
        else:
            standalone = False
        ax.fill_between(group_trend.index,
                        group_trend["mean"] - group_trend["sem"],
                        group_trend["mean"] + group_trend["sem"],
                        alpha=0.1, color="#028090")
        ax.plot(group_trend.index, group_trend["mean"],
                color="#028090", alpha=0.3, linewidth=2, label="Mean ± SEM")
        m_all, b_all, _, _, _ = stats.linregress(df_clean["Session order"],
                                                  df_clean["Retelling duration"])
        x_line = np.array([df_clean["Session order"].min(),
                           df_clean["Session order"].max()])

        if p_all > 0.05:
            line_color = "#808080"
            text_color = "#808080"
            fontweight = "normal"
        else:
            line_color = "black"
            text_color = "black"
            fontweight = "bold"

        ax.plot(x_line, b_all + m_all * x_line,
                color=line_color, linewidth=1.5, linestyle="--",
                label=f"Overall: r = {fmt_nolead(r_all)}, p {fmt_p(p_all)}")
        ax.set_xlabel("Trial order")
        ax.set_ylabel("Duration (s)")
        ax.text(x_line[-1], m_all * x_line[-1] + b_all + 0.75, f"r = {fmt_nolead(r_all)}", ha="right", va="bottom",
                fontsize=12, fontweight=fontweight, color=text_color)
        ax.text(x_line[-1], m_all * x_line[-1] + b_all - 0.25, f"p {fmt_p(p_all)}", ha="right", va="top",
                fontsize=12, fontweight=fontweight, color=text_color)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if standalone:
            fig.tight_layout()
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_trial_order.png", **SAVE_KW)
            plt.close(fig)


def trial_order_by_subject(n, df_clean, generate_figure=True):
    print_section_title(f"{n}. TRIAL ORDER EFFECT BY SUBJECT")

    print("Spearman correlation (session order vs retelling duration) per subject:")
    correlations = {}
    for subject, group in df_clean.groupby("Subject"):
        r, p = stats.spearmanr(group["Session order"], group["Retelling duration"])
        correlations[subject] = (r, p)
        if p < 0.05:
            sig = (Fore.BLUE + "\033[1m↗\033[0m" + Style.RESET_ALL if r > 0
                   else Fore.YELLOW + "\033[1m↘\033[0m" + Style.RESET_ALL)
        else:
            sig = Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
        print(f"  {subject}: r = {r:.3f}, p = {p:.4f} {sig}")

    if generate_figure:
        subjects = sorted(df_clean["Subject"].unique())
        subject_colors = {s: c for s, c in zip(subjects, sns.color_palette("tab20b", len(subjects)))}
        n_subj = len(subjects)
        n_cols = 6
        n_rows = math.ceil(n_subj / n_cols)

        fig, axes = plt.subplots(4, 6, figsize=(20, 14), sharey=True)
        axes = axes.flatten()
        for i, subject in enumerate(subjects):
            ax = axes[i]
            grp = df_clean[df_clean["Subject"] == subject].sort_values("Session order")
            color = subject_colors[subject]
            ax.scatter(grp["Session order"], grp["Retelling duration"],
                       color=color, alpha=0.5, s=25, zorder=3)
            m, b, r, p, _ = stats.linregress(grp["Session order"],
                                              grp["Retelling duration"])
            x_line = np.array([grp["Session order"].min(),
                                grp["Session order"].max()])
            lw = 2.5 if p < 0.05 else 1.5
            ls = "-" if p < 0.05 else "--"
            ax.plot(x_line, m * x_line + b, color=color, lw=lw, linestyle=ls)
            r_spear, p_spear = stats.spearmanr(grp["Session order"],
                                               grp["Retelling duration"])
            sig = "✓" if p_spear < 0.05 else "ns"
            ax.set_title(f"{subject}  (r = {r_spear:.2f}, {sig})", fontsize=10)
            ax.set_xlabel("Session order")
            ax.set_ylabel("Duration (s)" if i % n_cols == 0 else "")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Trial order vs retelling duration per subject (solid = p < .05)",
                     fontsize=13, y=1.01)
        fig.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_trial_order_by_subject.png", **SAVE_KW)
        plt.close(fig)


def retelling_duration_by_cartoon(n, df_clean, generate_figure=True):
    print_section_title(f"{n}. RETELLING DURATION BY CARTOON")

    by_cartoon = df_clean.groupby("Cartoon")["Retelling duration"].agg(["mean", "std", "min", "max", "count"])
    by_cartoon.columns = ["Mean (s)", "SD (s)", "Min (s)", "Max (s)", "N subjects"]
    by_cartoon = by_cartoon.round(2)

    print(f"Descriptive statistics per cartoon (N = {len(by_cartoon)} cartoons):")
    print(by_cartoon.to_string())

    print(f"\nSummary across cartoons:")
    print(f"  Shortest mean retelling : {by_cartoon['Mean (s)'].min():.2f} s"
          f"  ({by_cartoon['Mean (s)'].idxmin()})")
    print(f"  Longest  mean retelling : {by_cartoon['Mean (s)'].max():.2f} s"
          f"  ({by_cartoon['Mean (s)'].idxmax()})")
    print(f"  Grand mean (of means)   : {by_cartoon['Mean (s)'].mean():.2f} s"
          f"  (SD = {by_cartoon['Mean (s)'].std():.2f} s)")
    print(f"  Average N per cartoon   : {by_cartoon['N subjects'].mean():.1f}"
          f"  (range: {int(by_cartoon['N subjects'].min())}–{int(by_cartoon['N subjects'].max())})")

    # Kruskal-Wallis test
    groups = [grp["Retelling duration"].values for _, grp in df_clean.groupby("Cartoon")]
    k = len(groups)
    n_total = len(df_clean)

    h_stat, p_kruskal = stats.kruskal(*groups)
    eta2 = h_stat / (n_total - 1)

    sig_str = Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p_kruskal < 0.05 else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
    print(f"\nKruskal-Wallis: H({k - 1}) = {h_stat:.3f}, p = {p_kruskal:.4f}  {sig_str}")
    print(f"Effect size: η² = {eta2:.3f}")

    if p_kruskal < 0.05:
        print("\nThe retelling duration differs significantly across cartoons.")
    else:
        print("\nNo significant difference in retelling duration across cartoons.")

    # Pivot to subjects × cartoons matrix
    # cartoon_matrix = df_clean.pivot_table(index="Subject", columns="Cartoon",
    #                                       values="Retelling duration")
    # stat, p = stats.friedmanchisquare(*[cartoon_matrix[c].dropna() for c in cartoon_matrix.columns])
    # print(f"Friedman: χ²({len(cartoon_matrix.columns) - 1}) = {stat:.3f}, p = {p:.4f}")

    # LMM
    lmm = smf.mixedlm("Q('Retelling duration') ~ C(Cartoon)", df_clean,
                      groups=df_clean["Subject"]).fit()
    print(f"\n{'─' * 60}")
    print(f"  Mixed Linear Model — {"Retelling ~ Cartoon duration"}")
    print(f"{'─' * 60}")
    print(f"  {'Parameter':<35} {'β':>8} {'SE':>8} {'t':>8} {'p':>8}")
    print(f"  {'─' * 55}")
    for name, coef, se, t, p in zip(
            lmm.params.index,
            lmm.params,
            lmm.bse,
            lmm.tvalues,
            lmm.pvalues):
        if "Var" in name:
            continue
        sig = (Fore.GREEN + "✓" + Style.RESET_ALL if p < 0.05
               else Fore.RED + "✗" + Style.RESET_ALL)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:<35} {coef:>8.3f} {se:>8.3f} {t:>8.3f} {p:>8.4f}  {stars} {sig}")
    print(f"{'─' * 60}")
    print(f"  Random intercept variance : {lmm.cov_re.values[0][0]:.4f}")
    print(f"  Residual variance         : {lmm.scale:.4f}")
    icc = lmm.cov_re.values[0][0] / (lmm.cov_re.values[0][0] + lmm.scale)
    print(f"  ICC (subject)             : {icc:.4f}")
    print(f"  Log-likelihood            : {lmm.llf:.4f}")
    print(f"  AIC  : {2 * (len(lmm.params) - lmm.llf):.4f}")
    print(f"{'─' * 60}\n")

    if generate_figure:
        from matplotlib.patches import Patch
        import matplotlib.lines as mlines

        cartoon_show = df_clean.groupby("Cartoon")["Show"].first()
        plot_df = by_cartoon.copy()
        plot_df["Show"] = cartoon_show
        plot_df = plot_df.sort_values("Mean (s)")

        # Map each cartoon to its y-position after sorting
        cartoon_to_y = {cartoon: i for i, cartoon in enumerate(plot_df.index)}

        fig, ax = plt.subplots(figsize=(10, 20))
        y_pos = np.arange(len(plot_df))
        bar_colors = [SHOW_COLORS[s] for s in plot_df["Show"]]

        ax.barh(
            y_pos, plot_df["Mean (s)"],
            xerr=plot_df["SD (s)"],
            color=bar_colors, edgecolor="white", height=0.75,
            error_kw=dict(elinewidth=0.8, capsize=2, ecolor="gray", alpha=0.6),
        )

        subjects = df_clean["Subject"].unique()
        subject_colors = {s: c for s, c in zip(subjects, sns.color_palette("tab20b", len(subjects)))}

        # Individual data points per subject
        for subject, grp in df_clean.groupby("Subject"):
            ys = grp["Cartoon"].map(cartoon_to_y)
            ax.scatter(
                grp["Retelling duration"], ys,
                color=subject_colors[subject], s=18, zorder=3,
                alpha=0.85, linewidths=0.3, edgecolors="white",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df.index, fontsize=7)
        ax.set_xlabel("Mean retelling duration (s)")
        ax.set_title(
            f"Retelling duration by cartoon (N = {len(plot_df)}, sorted by mean)\n"
            f"Kruskal-Wallis: H({k - 1}) = {h_stat:.2f}, p = {p_kruskal:.4f}, η² = {eta2:.3f}"
        )

        # Legend: show colours (patches) + subject colours (circles)
        show_handles = [Patch(facecolor=SHOW_COLORS[s], label=s) for s in SHOW_COLORS]
        subject_handles = [
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor=subject_colors[s],
                          markersize=6, label=s)
            for s in sorted(subject_colors)
        ]
        ax.legend(
            handles=show_handles + subject_handles,
            title="Show / Subject", fontsize=9, title_fontsize=9,
            loc="lower right",
        )

        fig.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_duration_by_cartoon.png", **SAVE_KW)


def cartoon_duration_as_predictor(n, df_clean, generate_figure=True, ax=None):
    print_section_title(f"{n}. CARTOON DURATION AS PREDICTOR")

    # Overall correlation
    r, p = stats.pearsonr(df_clean["Cartoon duration"], df_clean["Retelling duration"])
    print(f"Overall Pearson correlation: r = {r:.3f}, p = {p:.4f}")

    # LMM: retelling ~ cartoon duration + (1|subject) — accounts for between-subject variance
    lmm = smf.mixedlm(
        "Q('Retelling duration') ~ Q('Cartoon duration')",
        df_clean,
        groups=df_clean["Subject"]
    ).fit(reml=True)
    lmm_coef = lmm.params["Q('Cartoon duration')"]
    lmm_t    = lmm.tvalues["Q('Cartoon duration')"]
    lmm_p    = lmm.pvalues["Q('Cartoon duration')"]
    lmm_se = lmm.bse["Q('Cartoon duration')"]
    lmm_ci   = lmm.conf_int().loc["Q('Cartoon duration')"]
    sym_lmm  = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if lmm_p < 0.05
                else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)
    print(f"\nLMM (retelling ~ cartoon duration + (1|subject)):")
    print(f"  β = {lmm_coef:.3f}, t = {lmm_t:.3f}, p = {lmm_p:.4f} {sym_lmm}")
    print(f"95% CI [{lmm_ci[0]:.4f}, {lmm_ci[1]:.4f}]")
    print(f"  Subject random intercept variance: {lmm.cov_re.values[0][0]:.4f}")

    # Linear regression
    result = pg.linear_regression(df_clean[["Cartoon duration"]], df_clean["Retelling duration"])
    print("\nLinear regression (retelling ~ cartoon duration):")
    print(result.to_string())

    # Per-subject Pearson correlations
    print("\nPearson correlation by subject:")
    subjects_order, r_values, p_values = [], [], []
    for subject, group in df_clean.groupby("Subject"):
        r_s, p_s = stats.pearsonr(group["Cartoon duration"], group["Retelling duration"])
        subjects_order.append(subject)
        r_values.append(r_s)
        p_values.append(p_s)

    from statsmodels.stats.multitest import multipletests
    _, p_fdr_values, _, _ = multipletests(p_values, method="fdr_bh")

    for subject, r_s, p_s, p_fdr in zip(subjects_order, r_values, p_values, p_fdr_values):
        if p_fdr < 0.05:
            sig = (Fore.BLUE + "\033[1m↗\033[0m" + Style.RESET_ALL if r_s > 0
                   else Fore.YELLOW + "\033[1m↘\033[0m" + Style.RESET_ALL)
        else:
            sig = Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
        print(f"  {subject}: r = {r_s:.3f}, p = {p_s:.4f}, p_FDR = {p_fdr:.4f} {sig}")

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

        subjects = df_clean["Subject"].unique()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            standalone = True
        else:
            standalone = False
        for subject in sorted(subjects):
            grp = df_clean[df_clean["Subject"] == subject]
            ax.scatter(grp["Cartoon duration"], grp["Retelling duration"],
                       color=SUBJECT_COLORS[subject], alpha=0.4, s=25, label="S" + subject[2:])
        m, b, r, p, _ = stats.linregress(df_clean["Cartoon duration"], df_clean["Retelling duration"])
        x_line = np.linspace(df_clean["Cartoon duration"].min(), df_clean["Cartoon duration"].max(), 200)

        if p > 0.05:
            line_color = "#808080"
            text_color = "#808080"
            fontweight = "normal"
        else:
            line_color = "black"
            text_color = "black"
            fontweight = "bold"

        ax.plot(x_line, m * x_line + b, color=line_color, lw=2, linestyle="--",
                label=f"Overall fit (r = {fmt_nolead(r)}, p {fmt_p(p)})")
        ax.set_xlabel("Cartoon duration (s)")
        ax.set_ylabel("Retelling duration (s)")
        ax.text(x_line[-1], m * x_line[-1] + b + 2.5, f"r = {fmt_nolead(r)}", ha="right", va="bottom",
                fontsize=12, fontweight=fontweight, color=text_color)
        ax.text(x_line[-1], m * x_line[-1] + b - 1.5, f"p {fmt_p(p)}", ha="right", va="top",
                fontsize=12, fontweight=fontweight, color=text_color)
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        if standalone:
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_cartoon_predictor.png", **SAVE_KW)
            plt.close(fig)


def retelling_duration_by_show(n, df_clean, generate_figure=True):
    print_section_title(f"{n}. RETELLING DURATION BY SHOW")
    test_duration_by_group(df_clean, "Show")

    if generate_figure:
        shows = df_clean["Show"].unique()
        fig, ax = plt.subplots(figsize=(8, 5))
        order_show = sorted(shows)
        palette_show = [SHOW_COLORS[s] for s in order_show]
        sns.boxplot(data=df_clean, x="Show", y="Retelling duration",
                    order=order_show, hue="Show", palette=SHOW_COLORS,
                    width=0.5, fliersize=0, legend=False, ax=ax)
        sns.stripplot(data=df_clean, x="Show", y="Retelling duration",
                      order=order_show, hue="Show", palette=SHOW_COLORS,
                      alpha=0.3, jitter=True, size=3, legend=False, ax=ax)
        ax.set_xlabel("Show")
        ax.set_ylabel("Retelling duration (s)")
        ax.set_title("Retelling duration by show (ns)")
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_duration_by_show.png", **SAVE_KW)

if __name__ == "__main__":
    # Load the Excel file and keep the relevant columns
    df = pd.read_excel(PATH_INFO)
    df_keep = df[df["Keep?"] == "Yes"]
    df_clean = df_keep[["Subject", "Show", "Cartoon", "Session order", "Retelling duration", "Cartoon duration"]].dropna()
    print(f"Total trials after cleaning: {len(df_clean)}\n")

    generate_figure = True
    overall_retelling_duration(1, df_clean, generate_figure)
    retelling_duration_by_subject(2, df_clean, generate_figure)
    trial_order(3, df_clean, generate_figure)
    trial_order_by_subject(4, df_clean, generate_figure)
    retelling_duration_by_cartoon(5, df_clean, generate_figure)
    cartoon_duration_as_predictor(6, df_clean, generate_figure)
    retelling_duration_by_show(7, df_clean, generate_figure)