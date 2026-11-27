import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy import stats
import pingouin as pg
from colorama import Fore, Back, Style
from constants import *
import os

import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.formula.api as smf

from helper_functions import save_figure, print_section_title, plt

PATH_FIGURES = op.join(PATH_PLOTS, "Duration")
os.makedirs(PATH_FIGURES, exist_ok=True)

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
        plt.close(fig)

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

    if generate_figure:
        order = df_clean.groupby("Subject")["Retelling duration"].mean().sort_values().index.tolist()
        anon_labels = {s: f"K{str(i+1).zfill(2)}" for i, s in enumerate(order)}

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))
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
        ax.set_xticklabels([anon_labels[s] for s in order], rotation=0)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Retelling duration (s)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        k06 = next(s for s in order if "K06" in s)
        sig_pairs = [
            (k06, next(s for s in order if "K02" in s)),
            (k06, next(s for s in order if "K03" in s)),
            (k06, next(s for s in order if "K05" in s)),
        ]

        y_max = df_clean["Retelling duration"].max()
        y_step = (df_clean["Retelling duration"].max() - df_clean["Retelling duration"].min()) * 0.06
        sig_pairs = sorted(sig_pairs, key=lambda p: abs(order.index(p[0]) - order.index(p[1])), reverse=True)
        for i, (s1, s2) in enumerate(sig_pairs):
            x1, x2 = order.index(s1), order.index(s2)
            y = y_max + y_step * (i + 1)
            ax.plot([x1, x1, x2, x2], [y, y + y_step * 0.4, y + y_step * 0.4, y],
                    color="black", lw=1.2)
            ax.text((x1 + x2) / 2, y + y_step * 0.45, "*", ha="center", va="bottom", fontsize=12)

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

    # LMM: retelling duration ~ session order + (1 | subject) — tests whether trial
    # order predicts duration after accounting for between-subject variance
    lmm = smf.mixedlm(
        "Q('Retelling duration') ~ Q('Session order')",
        df_clean,
        groups=df_clean["Subject"],
        re_formula="~ Q('Session order')"
    ).fit(reml=True)

    lmm_coef = lmm.params["Q('Session order')"]
    lmm_t = lmm.tvalues["Q('Session order')"]
    lmm_p = lmm.pvalues["Q('Session order')"]

    sym_lmm = (Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if lmm_p < 0.05
               else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL)

    print(f"\nLMM (retelling duration ~ session order + (session order|subject)):")
    print(f"  β = {lmm_coef:.3f}, t = {lmm_t:.3f}, p = {lmm_p:.4f} {sym_lmm}")

    print("\nRandom-effects covariance matrix:")
    print(lmm.cov_re)

    if generate_figure:
        def fmt_nolead(value):
            s = f"{value:.2f}"
            if s.startswith("0."):
                return s[1:]
            elif s.startswith("-0."):
                return "-" + s[2:]
            return s

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
                label=f"Overall: r = {fmt_nolead(r_all)}, p = {fmt_nolead(p_all)}")
        ax.set_xlabel("Trial order")
        ax.set_ylabel("Duration (s)")
        ax.text(x_line[-1], m_all * x_line[-1] + b_all + 0.5, f"r = {fmt_nolead(r_all)}", ha="right", va="bottom",
                fontsize=12, fontweight=fontweight, color=text_color)
        ax.text(x_line[-1], m_all * x_line[-1] + b_all - 0.5, f"p = {fmt_nolead(p_all)}", ha="right", va="top",
                fontsize=12, fontweight=fontweight, color=text_color)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if standalone:
            fig.tight_layout()
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_trial_order.png", **SAVE_KW)
            plt.close(fig)


def trial_order_by_subject(n, df_clean, generate_figure=True):
    print_section_title(f"{n}. TRIAL ORDER EFFECT BY SUBJECT")

    def fmt_nolead(value):
        s = f"{value:.2f}"
        if s.startswith("0."):
            return s[1:]
        elif s.startswith("-0."):
            return "-" + s[2:]
        return s

    print("Pearson correlation (session order vs retelling duration) by subject:")
    correlations = {}
    subjects_order, r_values, p_values = [], [], []
    for subject, group in df_clean.groupby("Subject"):
        r, p = stats.pearsonr(group["Session order"], group["Retelling duration"])
        correlations[subject] = (r, p)
        subjects_order.append(subject)
        r_values.append(r)
        p_values.append(p)

    from statsmodels.stats.multitest import multipletests
    _, p_fdr_values, _, _ = multipletests(p_values, method="fdr_bh")

    for subject, r, p, p_fdr in zip(subjects_order, r_values, p_values, p_fdr_values):
        correlations[subject] = (r, p, p_fdr)
        if p_fdr < 0.05:
            sig = (Fore.BLUE + "\033[1m↗\033[0m" + Style.RESET_ALL if r > 0
                   else Fore.YELLOW + "\033[1m↘\033[0m" + Style.RESET_ALL)
        else:
            sig = Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
        print(f"  {subject}: r = {r:.3f}, p = {p:.4f}, p_FDR = {p_fdr:.4f} {sig}")

    r_all, p_all = stats.pearsonr(df_clean["Session order"], df_clean["Retelling duration"])
    print(f"\nOverall: r = {r_all:.3f}, p = {p_all:.4f}")

    if generate_figure:
        subjects = df_clean["Subject"].unique()
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
        axes = axes.flatten()
        for i, subject in enumerate(sorted(subjects)):
            ax = axes[i]
            grp = df_clean[df_clean["Subject"] == subject].sort_values("Session order")
            color = SUBJECT_COLORS[subject]
            ax.scatter(grp["Session order"], grp["Retelling duration"],
                       color=color, alpha=0.5, s=25, zorder=3)
            m, b, r, p, _ = stats.linregress(grp["Session order"], grp["Retelling duration"])
            x_line = np.array([grp["Session order"].min(), grp["Session order"].max()])

            if p > 0.05:
                line_color = "#808080"
                text_color = "#808080"
                fontweight = "normal"
            else:
                line_color = "black"
                text_color = "black"
                fontweight = "bold"

            ax.plot(x_line, m * x_line + b, color=line_color,
                    lw=2.5 if p < 0.05 else 1.5, linestyle="-" if p < 0.05 else "--")
            ax.set_title(f"{subject}  (r = {fmt_nolead(r)}, p = {fmt_nolead(p)})",
                         fontsize=11, fontweight=fontweight, color=text_color)
            ax.set_xlabel("Session order")
            ax.set_ylabel("Duration (s)" if i % 3 == 0 else "")
        fig.suptitle("Trial order vs retelling duration (solid = p < .05)", fontsize=13, y=1.01)
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
    print("Note: with ~4 subjects per cartoon, this test is severely underpowered; "
          "interpret results with caution.")


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
            # xerr=plot_df["SD (s)"],
            color=bar_colors, edgecolor="white", height=0.75,
            # error_kw=dict(elinewidth=0.8, capsize=2, ecolor="gray", alpha=0.6),
        )

        # Individual data points per subject
        for subject, grp in df_clean.groupby("Subject"):
            ys = grp["Cartoon"].map(cartoon_to_y)
            ax.scatter(
                grp["Retelling duration"], ys,
                color=SUBJECT_COLORS[subject], s=18, zorder=3,
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
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor=SUBJECT_COLORS[s],
                          markersize=6, label=s)
            for s in sorted(SUBJECT_COLORS)
        ]
        ax.legend(
            handles=show_handles + subject_handles,
            title="Show / Subject", fontsize=9, title_fontsize=9,
            loc="lower right",
        )

        fig.tight_layout()
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_duration_by_cartoon.png", **SAVE_KW)
        plt.close(fig)


def cartoon_duration_as_predictor(n, df_clean, generate_figure=True, ax=None):
    print_section_title(f"{n}. CARTOON DURATION AS PREDICTOR")

    import statsmodels.formula.api as smf

    # Pooled Pearson (ignores subject structure, reported for reference)
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

    from scipy.stats import kruskal

    # Bin continuous predictors into groups for Kruskal-Wallis
    df_clean["Order bin"] = pd.qcut(df_clean["Session order"], q=4)
    df_clean["Duration bin"] = pd.qcut(df_clean["Cartoon duration"], q=4)

    for col, label in [("Order bin", "Session order"), ("Duration bin", "Cartoon duration"), ("Cartoon", "Cartoon")]:
        groups = [g["Retelling duration"].values for _, g in df_clean.groupby(col)]
        H, p = kruskal(*groups)
        eta2 = (H - len(groups) + 1) / (len(df_clean) - len(groups))
        print(f"{label}: H = {H:.3f}, p = {p:.4f}, η² = {eta2:.3f}")

    groups = [g["Retelling duration"].values for _, g in df_clean.groupby("Subject")]
    H, p = kruskal(*groups)
    eta2 = (H - len(groups) + 1) / (len(df_clean) - len(groups))
    print(f"Subject: H = {H:.3f}, p = {p:.4f}, η² = {eta2:.3f}")

    # Per-subject Pearson correlations
    print("\nPearson correlation by subject:")
    for subject, group in df_clean.groupby("Subject"):
        r_s, p_s = stats.pearsonr(group["Cartoon duration"], group["Retelling duration"])
        if p_s < 0.05:
            sig = (Fore.BLUE + "\033[1m↗\033[0m" + Style.RESET_ALL if r_s > 0
                   else Fore.YELLOW + "\033[1m↘\033[0m" + Style.RESET_ALL)
        else:
            sig = Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
        print(f"  {subject}: r = {r_s:.3f}, p = {p_s:.4f} {sig}")

    if generate_figure:
        subjects = df_clean["Subject"].unique()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            standalone = True
        else:
            standalone = False
        for subject in sorted(subjects):
            grp = df_clean[df_clean["Subject"] == subject]
            ax.scatter(grp["Cartoon duration"], grp["Retelling duration"], color=SUBJECT_COLORS[subject], alpha=0.4,
                       s=25, label=subject)
        m, b, r, p, _ = stats.linregress(df_clean["Cartoon duration"], df_clean["Retelling duration"])
        x_line = np.linspace(df_clean["Cartoon duration"].min(), df_clean["Cartoon duration"].max(), 200)

        def fmt_nolead(value):
            s = f"{value:.2f}"
            if s.startswith("0."):
                return s[1:]
            elif s.startswith("-0."):
                return "-" + s[2:]
            return s

        if p > 0.05:
            line_color = "#808080"
            text_color = "#808080"
            fontweight = "normal"
        else:
            line_color = "black"
            text_color = "black"
            fontweight = "bold"

        ax.plot(x_line, m * x_line + b, color=line_color, lw=2, linestyle="--",
                label=f"Overall fit (r = {fmt_nolead(r)}, p = {fmt_nolead(p)})")
        ax.set_xlabel("Cartoon duration (s)")
        ax.set_ylabel("Retelling duration (s)")
        ax.text(x_line[-1], m * x_line[-1] + b + 0.5, f"r = {fmt_nolead(r)}", ha="right", va="bottom", fontsize=12,
                fontweight=fontweight, color=text_color)
        ax.text(x_line[-1], m * x_line[-1] + b - 0.5, f"p = {fmt_nolead(p)}", ha="right", va="top", fontsize=12,
                fontweight=fontweight, color=text_color)
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
        plt.close(fig)


def mask_condition(n, df_clean, generate_figure=True, ax=None, p_for_plot=None, p_label_override=None):
    print_section_title(f"{n}. MASK CONDITION")
    by_condition = df_clean.groupby("Masked?")["Retelling duration"].agg(["mean", "std", "min", "max", "count"])
    by_condition.columns = ["Mean (s)", "SD (s)", "Min (s)", "Max (s)", "N trials"]
    by_condition = by_condition.round(2)
    print(by_condition.to_string())

    # Per-subject means per condition
    subject_condition_means = df_clean.groupby(["Subject", "Masked?"])["Retelling duration"].mean().unstack("Masked?")
    print(f"\nSubject means per condition:")
    print(subject_condition_means.round(2).to_string())

    p_main = None
    conditions = subject_condition_means.columns.tolist()
    if len(conditions) == 2:
        a, b = conditions
        diff_scores = subject_condition_means[a] - subject_condition_means[b]

        # Step 1: Normality on difference scores
        stat, p_norm = stats.shapiro(diff_scores)
        normal = p_norm > 0.05
        print(
            f"\nShapiro-Wilk on difference scores ({a} - {b}): W = {stat:.3f}, p = {p_norm:.4f} "
            f"{Fore.GREEN + '\033[1m✓\033[0m' if normal else Fore.RED + '\033[1m✗\033[0m'}",
            end="")
        print(Style.RESET_ALL)
        if normal:
            print("Difference scores follow a normal distribution.")
        else:
            print("Difference scores do not follow a normal distribution.")

        # Step 2: Main test
        if normal:
            print("\nAs the difference scores " + Fore.GREEN + "follow a normal distribution" + Style.RESET_ALL +
                  ", we perform a " + "\033[1m" + "paired t-test." + "\033[0m\n")
            t, p_main = stats.ttest_rel(subject_condition_means[a], subject_condition_means[b])
            d = pg.compute_effsize(subject_condition_means[a], subject_condition_means[b], paired=True,
                                   eftype="cohen")
            power = pg.power_ttest(d=abs(d), n=len(subject_condition_means), alpha=0.05, contrast="paired")
            print(f"  Mean difference : {diff_scores.mean():.2f} s (SD = {diff_scores.std():.2f} s)")
            print(f"  t({len(subject_condition_means) - 1}) = {t:.3f}, p = {p_main:.4f}")
            print(f"  Cohen's d = {d:.3f}")
            print(f"  Power     = {power:.3f}")
        else:
            print("\nAs the difference scores " + Fore.RED + "do not follow a normal distribution" + Style.RESET_ALL +
                  ", we perform a " + "\033[1m" + "Wilcoxon signed-rank test." + "\033[0m\n")
            w_stat, p_main = stats.wilcoxon(subject_condition_means[a], subject_condition_means[b])
            r = pg.compute_effsize(subject_condition_means[a], subject_condition_means[b], paired=True, eftype="r")
            print(f"  Median difference : {diff_scores.median():.2f} s")
            print(f"  W = {w_stat:.3f}, p = {p_main:.4f}")
            print(f"  Effect size r = {r:.3f}")

        # Step 3: Wilcoxon as non-parametric confirmation if parametric was used
        if normal:
            w_stat, p_wilcoxon = stats.wilcoxon(subject_condition_means[a], subject_condition_means[b])
            print(f"\nWilcoxon signed-rank (non-parametric confirmation): W = {w_stat:.3f}, p = {p_wilcoxon:.4f}")
    else:
        print(f"Warning: expected 2 conditions, found {len(conditions)}: {conditions}")

    if generate_figure:
        from scipy.stats import t as t_dist
        from matplotlib.lines import Line2D

        subject_means = subject_condition_means
        conditions = ["Masked", "Unmasked"]
        colors = {c: CONDITION_COLORS[c] for c in conditions}

        # Block 1 = did that condition first
        condition_block = df_clean.groupby(["Subject", "Masked?"])["Block"].first()
        masked_first = {subject: condition_block[subject]["Masked"] == 1 for subject in subject_means.index}

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            standalone = True
        else:
            standalone = False
        x = np.arange(len(conditions))
        offset_mean = 0.12  # Horizontal distance between the subject dots and the group mean diamond

        # Paired subject lines and dots: solid if the masked condition was presented first, dashed otherwise
        for subject in sorted(subject_means.index):
            row = subject_means.loc[subject].astype(float)
            if row[conditions].isna().any():
                continue
            linestyle = "-" if masked_first[subject] else "--"
            ax.plot(x, [row["Masked"], row["Unmasked"]], color="#a0a0a0", lw=1.2, ls=linestyle, zorder=2)
            ax.scatter(x, [row["Masked"], row["Unmasked"]], color="#60606080", edgecolors="#30303080", s=50,
                       zorder=3)

        # Group mean ± 95% CI, next to the dots, on the outer side
        ci_tops, x_means = [], []
        for i, cond in enumerate(conditions):
            vals = subject_means[cond].dropna().astype(float).values
            m, sem = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
            ci = sem * t_dist.ppf(0.975, len(vals) - 1)
            ci_tops.append(m + ci)
            x_pos = x[i] - offset_mean if i == 0 else x[i] + offset_mean
            x_means.append(x_pos)
            ax.errorbar(x_pos, m, yerr=ci, fmt="D", color=darken(colors[cond]), mfc=colors[cond],
                        mec=darken(colors[cond]), ms=8, elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

        # Legend for the presentation order
        handles = [Line2D([0], [0], color="#a0a0a0", lw=1.2, ls="-", label="Masked first"),
                   Line2D([0], [0], color="#a0a0a0", lw=1.2, ls="--", label="Unmasked first")]
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2)

        ax.set_xlim(x[0] - 0.5, x[1] + 0.5)

        # Significance bracket (placed above both the data and the CIs), label taken from the test performed
        y_max = max(subject_means[conditions].astype(float).max().max(), max(ci_tops)) * 1.01
        ax.plot([x_means[0], x_means[0], x_means[1], x_means[1]],
                [y_max, y_max * 1.005, y_max * 1.005, y_max], color="#808080", lw=1.1)
        p_plot = p_for_plot if p_for_plot is not None else p_main
        if p_label_override is not None:
            p_label = p_label_override
        elif p_plot is None or p_plot >= 0.05:
            p_label = "n.s."
        else:
            p_label = "***" if p_plot < 0.001 else "**" if p_plot < 0.01 else "*"
        ax.text(np.mean(x), y_max * 1.01, p_label, ha="center", va="bottom", fontsize=11)
        ax.set_ylim(top=y_max * 1.04)  # Leave room for the significance label inside the axes

        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel("Mean retelling duration (s)")
        if standalone:
            fig.tight_layout()
            save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_mask_condition.png", **SAVE_KW)
            plt.close(fig)

def subject_x_mask_interaction(n, df_clean, generate_figure=True):
    print_section_title(f"{n}. SUBJECT × MASK INTERACTION")

    print("Mask effect per subject (independent t-test: Masked vs Unmasked trials):")
    print(f"  {'Subject':<15} {'Masked (s)':<12} {'Unmasked (s)':<14} {'Diff (s)':<10} {'t':<8} {'p':<8} {'d':<6}")
    print("  " + "-" * 70)

    for subject, group in df_clean.groupby("Subject"):
        masked = group[group["Masked?"] == "Masked"]["Retelling duration"]
        unmasked = group[group["Masked?"] == "Unmasked"]["Retelling duration"]
        t, p = stats.ttest_ind(unmasked, masked)
        d = pg.compute_effsize(unmasked, masked, paired=False, eftype="cohen")
        diff = unmasked.mean() - masked.mean()
        sig = Fore.GREEN + "\033[1m✓\033[0m" + Style.RESET_ALL if p < 0.05 else Fore.RED + "\033[1m✗\033[0m" + Style.RESET_ALL
        print(f"  {subject:<15} {unmasked.mean():<14.2f} {masked.mean():<12.2f} "
              f"{diff:<10.2f} {t:<8.3f} {p:<8.4f} {d:.3f} {sig}")

    if generate_figure:
        subjects = df_clean["Subject"].unique()
        fig, ax = plt.subplots(figsize=(10, 5))
        # Boxplot side by side per subject
        df_plot = df_clean.copy()
        subject_labels = {s: s for s in subjects}
        df_plot["SubjectShort"] = df_plot["Subject"].map(subject_labels)
        order_short = [s for s in sorted(subjects)]
        sns.boxplot(data=df_plot, x="SubjectShort", y="Retelling duration", hue="Masked?",
                    order=order_short, hue_order=["Unmasked", "Masked"],
                    palette=CONDITION_COLORS, width=0.6, fliersize=0, ax=ax)
        sns.stripplot(data=df_plot, x="SubjectShort", y="Retelling duration", hue="Masked?",
                      order=order_short, hue_order=["Unmasked", "Masked"],
                      palette=CONDITION_COLORS, alpha=0.3, jitter=True, size=3,
                      dodge=True, ax=ax, legend=False)
        # Significance markers (p < .05)
        sig_subjects = {"K02": 0.0000, "K03": 0.0253, "K05": 0.0330, "K06": 0.0001}
        subject_max = df_plot.groupby("SubjectShort")["Retelling duration"].max()
        for i, subj in enumerate(order_short):
            if subj in sig_subjects:
                y_top = subject_max[subj] + 1
                ax.text(i, y_top, "*", ha="center", va="bottom", fontsize=16, fontweight="bold")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Retelling duration (s)")
        ax.set_title("Subject × Mask interaction  (* p < .05)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Condition")
        save_figure(fig, PATH_FIGURES, f"fig{str(n).zfill(2)}_subject_mask_interaction.png", **SAVE_KW)
        plt.close(fig)


if __name__ == "__main__":
    # Load the Excel file and keep the relevant columns

    def is_excluded(row):
        return row["Cartoon"] in EXCLUDED_RECORDINGS.get(row["Subject"], [])

    df = pd.read_excel(PATH_INFO, "Videos (excl.)")
    df_clean = df[["Subject", "Show", "Cartoon", "Block", "Session order", "Masked?", "Retelling duration",
                   "Cartoon duration"]].dropna()
    print(f"Total trials before cleaning: {len(df_clean)}\n")
    df_clean = df_clean[~df_clean.apply(is_excluded, axis=1)]
    print(f"Total trials after cleaning: {len(df_clean)}\n")

    generate_figure = True
    overall_retelling_duration(1, df_clean, generate_figure)
    retelling_duration_by_subject(2, df_clean, generate_figure)
    trial_order(3, df_clean, generate_figure)
    trial_order_by_subject(4, df_clean, generate_figure)
    retelling_duration_by_cartoon(5, df_clean, generate_figure)
    cartoon_duration_as_predictor(6, df_clean, generate_figure)
    retelling_duration_by_show(7, df_clean, generate_figure)
    mask_condition(8, df_clean, generate_figure)
    subject_x_mask_interaction(9, df_clean, generate_figure)