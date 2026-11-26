import matplotlib
matplotlib.use("TkAgg")

from itertools import product

from constants import *
from krajjat import Results
from krajjat.plot_functions import *
from krajjat.classes.graph_element import Graph

import seaborn as sns

from constants import *
from krajjat import analysis_functions as afunc
import pandas as pd

from matplotlib import font_manager
from pathlib import Path
for path in Path(r"Metropolis").glob("*.otf"):
    font_manager.fontManager.addfont(str(path))
for path in Path(r"Merriweather").glob("*.otf"):
    font_manager.fontManager.addfont(str(path))

SAVE_KW = dict(dpi=150, bbox_inches="tight")

AVERAGE_COLOR = "#1A1A1A"   # near-black line for the cross-subject average

def _annotate_reref_axis(fig, joint_name="SpineBase", hatch="////",
                         text="Re-referencing joint"):
    import matplotlib.patches as mpatches

    for ax in fig.axes:
        if ax.get_title() == joint_name:
            # diagonal hatch filling the whole axis
            rect = mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                hatch=hatch,
                facecolor="white",
                edgecolor="lightgray",
                linewidth=0.5,
                zorder=0,
            )
            ax.add_patch(rect)
            # italic label centred on the axis
            ax.text(
                0.5, 0.5, text,
                transform=ax.transAxes,
                ha="center", va="center",
                fontstyle="italic", fontsize=7,
                color="gray", zorder=2,
            )
            break


# ── File loading ──────────────────────────────────────────────────────────────
def _load_coherence_file(kinematic_measure, coherence_with, level, re_referenced, average):
    subject_trail = ""
    average_trail = ""
    if coherence_with in DERIVATIVES:
        if level == "group":
            subpath = op.join("Audio", "Group")
        else:
            subpath = op.join("Audio", "Per subject", f"{kinematic_measure} x {coherence_with}")
            subject_trail = "_" + level
            average_trail = ""
            if average == "trial":
                average_trail = "_avgtrial"
    else:
        if level == "group":
            subpath = "Joint"
        else:
            raise ValueError("If coherence_with is a joint, level must be 'group'.")

    reref_trail = REREF_SUFFIX if re_referenced else ""
    path_coherence_result = op.join(PATH_RESULTS, "Coherence", subpath,
                                    f"coherence_{kinematic_measure}_{coherence_with}"
                                    f"{subject_trail}{average_trail}{reref_trail}.gzip")
    print(f"Loading file {path_coherence_result}")
    return Results.load(path_coherence_result)


def _load_coherence_files(kinematic_measure, coherence_with, level, re_referenced, average):
    if isinstance(level, list):
        return [(subj, _load_coherence_file(kinematic_measure, coherence_with, subj, re_referenced, average))
                for subj in level]
    label = "Group" if level == "group" else level
    return [(label, _load_coherence_file(kinematic_measure, coherence_with, level, re_referenced, average))]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _anon(label):
    """Return the anonymised display name for a subject label."""
    return SUBJECT_ANON.get(label, label)


def _print_results_summary(results, kinematic_measure, coherence_with):
    if coherence_with in DERIVATIVES:
        txt_coherence_with = f"the {coherence_with} of the audio"
    else:
        txt_coherence_with = f"the {kinematic_measure} of the {coherence_with} joint"
    _, first_results = results[0]
    title = (f"Coherence between the {kinematic_measure} of "
             f"{len(first_results.labels_mocap)} joint(s) and {txt_coherence_with}")
    sep = "=" * len(title)
    print("=" * (len(title) // 2 - 5) + " SUMMARY " + "=" * (len(title) // 2 - (4 if len(title) % 2 == 0 else 3)))
    print(title)
    print(sep)
    print(f"\tSubjects: {len(results)}")
    print(f"\tSampling rate of the data: {first_results.sampling_rate} Hz")
    print(f"\tLags: {len(first_results.lags)} "
          f"({round(first_results.lags[0], 3)} s to {round(first_results.lags[-1], 3)} s; "
          f"step: {round(first_results.lags[1] - first_results.lags[0], 3)} s)")
    print(f"\tFrequencies: {round(first_results.frequencies[0], 3)} Hz to "
          f"{round(first_results.frequencies[-1], 3)} Hz "
          f"(step: {round(first_results.frequencies[1] - first_results.frequencies[0], 3)} Hz)")
    print(f"\tNumber of random permutations: {first_results.number_of_randperms}")
    if first_results.permutation_method is not None:
        print(f"\tPermutation method: {first_results.permutation_method.title()}")
    return title


def _get_closest_lags(results, lags):
    if lags is None:
        lags = results.lags
    if np.isscalar(lags):
        lags = [lags]
    return [results.lags[int(np.argmin(np.abs(np.array(results.lags) - l)))] for l in lags]


def _get_closest_frequencies(results, frequencies):
    freqs = np.array(results.frequencies)
    if frequencies == "all":
        return freqs
    if np.isscalar(frequencies):
        frequencies = [frequencies]
    return [freqs[int(np.argmin(np.abs(freqs - f)))] for f in frequencies]


# ── Normalization ─────────────────────────────────────────────────────────────
def _fit_fooof(freqs, values, freq_range=(0.25, 10.0)):
    from fooof import FOOOF
    freqs = np.array(freqs, dtype=float)
    values = np.array(values, dtype=float)
    range_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if range_mask.sum() < 4:
        return np.zeros_like(values), None
    freqs_fit = freqs[range_mask]
    values_fit = values[range_mask]
    if not np.all(values_fit > 0):
        return np.zeros_like(values), None
    fm = FOOOF(peak_width_limits=[0.5, 4.0], max_n_peaks=4, aperiodic_mode="fixed", verbose=False)
    fm.fit(freqs_fit, values_fit)
    residual = np.zeros_like(values)
    residual[range_mask] = np.log10(values_fit) - fm._ap_fit
    return residual, fm


def _apply_normalization(to_plot, sd, freqs, normalization):
    freqs = np.array(freqs)
    to_plot = np.array(to_plot, dtype=float)
    if sd is not None:
        sd = np.array(sd, dtype=float)

    if normalization is None:
        pass

    elif normalization == "*f":
        to_plot = to_plot * freqs
        if sd is not None:
            sd = sd * freqs

    elif normalization == "max":
        max_val = np.nanmax(to_plot)
        if max_val > 0:
            to_plot = to_plot / max_val
            if sd is not None:
                sd = sd / max_val

    elif normalization == "subject_peak":
        # Normalise to each subject's own spectral peak — makes all subjects peak at 1.
        peak_val = np.nanmax(np.abs(to_plot))
        if peak_val > 0:
            to_plot = to_plot / peak_val
            if sd is not None:
                sd = sd / peak_val

    elif normalization == "fooof":
        if np.any(np.array(to_plot) < 0):
            print("  Warning: 'fooof' normalization requires positive values (use kind='average'). "
                  "Falling back to '*f' normalization.")
            to_plot = to_plot * freqs
            if sd is not None:
                sd = sd * freqs
        else:
            residual, fm = _fit_fooof(freqs, to_plot)
            if fm is None:
                print("  Warning: FOOOF fit failed (insufficient positive values), returning raw values.")
            else:
                to_plot = residual
                if sd is not None:
                    background = np.zeros_like(freqs)
                    range_mask = (freqs >= 0.25) & (freqs <= 10.0)
                    background[range_mask] = 10 ** fm._ap_fit
                    with np.errstate(invalid="ignore", divide="ignore"):
                        scale = np.where(background > 0, np.abs(residual / background), 1.0)
                    sd = sd * scale

    else:
        print(f"  Warning: unknown normalization '{normalization}', using raw values.")

    return to_plot, sd


# ── z-score helper ────────────────────────────────────────────────────────────
def _compute_z_new(results, target_measure, kinematic_measure, joint, lag):
    avg = np.array(results.averages[target_measure][kinematic_measure]["Full dataset"][joint][lag])
    stds = np.array(results.stds[target_measure][kinematic_measure]["Full dataset"][joint][lag])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(stds > 0, avg / stds, np.nan)


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _get_curve(results, kind, target_measure, kinematic_measure, joint, lag):
    """Return (to_plot, sd) for a single subject/joint/lag combination."""
    if kind in ["z-scores", "z"]:
        return results.z_scores[target_measure][kinematic_measure]["Full dataset"][joint][lag], None
    elif kind == "z-new":
        return _compute_z_new(results, target_measure, kinematic_measure, joint, lag), None
    else:
        to_plot = results.averages[target_measure][kinematic_measure]["Full dataset"][joint][lag]
        sd = results.stds[target_measure][kinematic_measure]["Full dataset"][joint][lag]
        return to_plot, sd


def _add_average_to_graph(graph, curves_per_lag, freqs_or_lags_ms, lags_or_freqs, is_lags=False,
                           show_sd=False, error_type="standard_deviation"):
    for key, curve_list in curves_per_lag.items():
        if not curve_list:
            continue
        stacked = np.vstack(curve_list)
        if np.all(np.isnan(stacked)):
            continue
        avg_curve = np.nanmean(stacked, axis=0)
        if show_sd in [True, "average"]:
            n = np.sum(~np.isnan(stacked), axis=0)
            err_curve = np.nanstd(stacked, axis=0)
            if error_type == "standard_error":
                err_curve = err_curve / np.sqrt(np.maximum(n, 1))
        else:
            err_curve = None
        if is_lags:
            plot_label = f"Mean — {round(key, 3)} Hz" if len(curves_per_lag) > 1 else "Mean"
        else:
            plot_label = f"Mean — {int(key * 1000)} ms" if len(curves_per_lag) > 1 else "Mean"
        graph.add_plot(freqs_or_lags_ms, avg_curve, err_curve, 2.0,
                       color=AVERAGE_COLOR, label=plot_label, line_style="-")


# ── Body-graph plot ───────────────────────────────────────────────────────────
def _plot_body_graph(results_list, kind, kinematic_measure, lags, normalization,
                     show_average, title, show, path_png, **kwargs):
    plot_dictionary = {}
    multi_subject = len(results_list) > 1
    line_styles = ["-", "--", "-.", ":"]
    _, first_results = results_list[0]
    target_measure = list(first_results.target_measures)[0]
    freqs = np.array(first_results.frequencies)
    joints = first_results.labels_mocap

    for joint in joints:
        graph = Graph()
        # Collect curves per lag for the average
        curves_per_lag = {lag: [] for lag in lags}

        # Gather all subject curves first so we can prepend the average if needed
        subject_entries = []
        for s_idx, (label, results) in enumerate(results_list):
            for j, lag in enumerate(lags):
                to_plot, sd = _get_curve(results, kind, target_measure, kinematic_measure, joint, lag)
                to_plot, sd = _apply_normalization(to_plot, sd, freqs, normalization)
                curves_per_lag[lag].append(np.array(to_plot, dtype=float))
                color = SUBJECT_COLORS.get(label, "#888888")
                style = line_styles[j % len(line_styles)] if multi_subject else "-"
                display_name = _anon(label)
                plot_label = (f"{display_name} — {int(lag * 1000)} ms"
                              if (multi_subject and len(lags) > 1) else display_name)
                subject_entries.append((freqs, to_plot, sd, color, style, plot_label))

        if show_average == "above":
            _add_average_to_graph(graph, curves_per_lag, freqs, lags)
        for freqs_v, to_plot, sd, color, style, plot_label in subject_entries:
            graph.add_plot(freqs_v, to_plot, sd, 1, color=color, label=plot_label, line_style=style)
        if show_average == "below":
            _add_average_to_graph(graph, curves_per_lag, freqs, lags)

        plot_dictionary[joint] = graph

    fig = plot_body_graphs(plot_dictionary, title=None, show=show, path_save=None, close_fig=False, **kwargs)
    _annotate_reref_axis(fig)
    if path_png:
        fig.savefig(path_png)
        print(f"\tSaved the figure: {path_png}")
    plt.close(fig)


# ── Lag plot ──────────────────────────────────────────────────────────────────
def _plot_lags(results_list, kind, kinematic_measure, frequencies, show_average,
               title, show, path_png, show_sd=False, error_type="standard_deviation",
               show_peaks=False, show_peaks_sd=False, **kwargs):
    _, first_results = results_list[0]
    target_measure = list(first_results.target_measures)[0]
    freq_indices = [int(np.argmin(np.abs(np.array(first_results.frequencies) - f))) for f in frequencies]
    lags = first_results.lags
    # Reverse sign convention: negative lag = gesture precedes speech (matching cross-correlation)
    # lags_reversed = lags[::-1]
    lags_reversed = lags
    lags_ms = np.array(lags_reversed) * 1000  # ascending, negative = gesture ahead
    multi_subject = len(results_list) > 1
    line_styles = ["-", "--", "-.", ":"]
    joints = first_results.labels_mocap
    plot_dictionary = {}
    # peak_info[joint][freq] = list of (x_ms, y, color) per subject
    peak_info = {joint: {f: [] for f in frequencies} for joint in joints}
    # avg_curves[joint][freq] = cross-subject mean curve (for errorbar anchor)
    avg_curves = {joint: {} for joint in joints}

    for joint in joints:
        graph = Graph()
        curves_per_freq = {f: [] for f in frequencies}

        subject_entries = []
        for s_idx, (label, results) in enumerate(results_list):
            for f_idx, fi in enumerate(freq_indices):
                if kind in ["z-scores", "z"]:
                    to_plot = np.array(
                        [results.z_scores[target_measure][kinematic_measure]["Full dataset"][joint][lag][fi]
                         for lag in lags_reversed], dtype=float)
                    sd = None
                elif kind == "z-new":
                    to_plot = np.array(
                        [_compute_z_new(results, target_measure, kinematic_measure, joint, lag)[fi]
                         for lag in lags_reversed], dtype=float)
                    sd = None
                else:
                    to_plot = np.array(
                        [results.averages[target_measure][kinematic_measure]["Full dataset"][joint][lag][fi]
                         for lag in lags_reversed], dtype=float)
                    sd = np.array(
                        [results.stds[target_measure][kinematic_measure]["Full dataset"][joint][lag][fi]
                         for lag in lags_reversed], dtype=float)

                curves_per_freq[frequencies[f_idx]].append(np.array(to_plot, dtype=float))
                color = SUBJECT_COLORS.get(label, "#888888")
                style = line_styles[f_idx % len(line_styles)] if multi_subject else "-"
                display_name = _anon(label)
                plot_label = (f"{display_name} — {round(frequencies[f_idx], 3)} Hz"
                              if (multi_subject and len(frequencies) > 1) else display_name)
                sd_to_pass = sd if show_sd in [True, "subject"] else None
                subject_entries.append((lags_ms, to_plot, sd_to_pass, color, style, plot_label))

                if (show_peaks or show_peaks_sd) and np.any(np.isfinite(to_plot)):
                    peak_idx = int(np.nanargmax(to_plot))
                    peak_info[joint][frequencies[f_idx]].append((lags_ms[peak_idx], to_plot[peak_idx], color))

        if show_peaks_sd:
            for freq, curve_list in curves_per_freq.items():
                if curve_list:
                    avg_curves[joint][freq] = np.nanmean(np.vstack(curve_list), axis=0)

        if show_average == "above":
            _add_average_to_graph(graph, curves_per_freq, lags_ms, frequencies, is_lags=True,
                                  show_sd=show_sd, error_type=error_type)
        for lags_v, to_plot, sd, color, style, plot_label in subject_entries:
            graph.add_plot(lags_v, to_plot, sd, 1, color=color, label=plot_label, line_style=style)
        if show_average == "below":
            _add_average_to_graph(graph, curves_per_freq, lags_ms, frequencies, is_lags=True,
                                  show_sd=show_sd, error_type=error_type)

        plot_dictionary[joint] = graph

    if show_sd is not False:
        kwargs.setdefault("shaded_error_bars", True)
    fig = plot_body_graphs(plot_dictionary, title=None, show=False, path_save=None, close_fig=False, **kwargs)
    _annotate_reref_axis(fig)

    if show_peaks or show_peaks_sd:
        all_titles = [ax.get_title() for ax in fig.axes]
        matched = [t for t in all_titles if t in peak_info]
        print(f"  [peaks] joints in peak_info : {sorted(peak_info.keys())}")
        print(f"  [peaks] all axis titles      : {all_titles}")
        print(f"  [peaks] matched              : {matched}")
        print(f"  [peaks] entries per joint    : { {j: {f: len(e) for f, e in fd.items()} for j, fd in peak_info.items()} }")
        for ax in fig.axes:
            joint = ax.get_title()
            if joint not in peak_info:
                continue
            for freq, entries in peak_info[joint].items():
                if not entries:
                    continue
                xs = np.array([e[0] for e in entries])
                ys = np.array([e[1] for e in entries])
                colors = [e[2] for e in entries]
                if show_peaks:
                    ax.scatter(xs, ys, s=30, color=colors, zorder=5, linewidths=0.5, edgecolors="white")
                if show_peaks_sd:
                    avg = avg_curves[joint].get(freq)
                    if avg is None or not np.any(np.isfinite(avg)):
                        continue
                    peak_idx = int(np.nanargmax(avg))
                    anchor_x = float(lags_ms[peak_idx])
                    anchor_y = float(avg[peak_idx])
                    xs_finite = xs[np.isfinite(xs)]
                    n = len(xs_finite)
                    x_err = float(np.nanstd(xs))
                    if error_type == "standard_error":
                        x_err = x_err / np.sqrt(max(n, 1))
                    ax.errorbar(anchor_x, anchor_y, xerr=x_err, fmt="none",
                                color=AVERAGE_COLOR, linewidth=1.5, capsize=4, zorder=6)

    if show:
        plt.show()
    if path_png:
        fig.savefig(path_png)
        print(f"\tSaved the figure: {path_png}")

    plt.close(fig)


# ── Silhouette plot ───────────────────────────────────────────────────────────
def _plot_silhouette(results_list, kind, kinematic_measure, lags, frequencies,
                     title, show, path_png, **kwargs):
    _, first_results = results_list[0]
    target_measure = list(first_results.z_scores.keys())[0]
    freq_indices = [int(np.argmin(np.abs(np.array(first_results.frequencies) - f))) for f in frequencies]
    joints = list(first_results.z_scores[target_measure][kinematic_measure]["Full dataset"].keys())
    plot_dictionary = {joint: [] for joint in joints}
    combos = list(product(results_list, lags, list(zip(freq_indices, frequencies))))

    for joint in joints:
        for (label, results), lag, (fi, fa) in combos:
            tm = list(results.z_scores.keys())[0]
            if kind in ["z-scores", "z"]:
                value = results.z_scores[tm][kinematic_measure]["Full dataset"][joint][lag][fi]
            elif kind == "z-new":
                value = float(_compute_z_new(results, tm, kinematic_measure, joint, lag)[fi])
            else:
                value = results.averages[tm][kinematic_measure]["Full dataset"][joint][lag][fi]
            plot_dictionary[joint].append(float(value) if np.isfinite(value) else 0.0)

    titles = [f"{int(lag * 1000)} ms\n{fa:.2f} Hz" for (label, _), lag, (_, fa) in combos]
    plot_silhouette(plot_dictionary, title=title,
                    title_silhouette=titles if len(combos) > 1 else titles[0],
                    title_scale=kind, color_scheme="celsius", show=show, path_save=path_png, **kwargs)
    if path_png:
        print(f"\tSaved the figure: {path_png}")


# ── Public entry point ────────────────────────────────────────────────────────
def plot_coherence(kinematic_measure, coherence_with, kind="z-scores", level="group", re_referenced=True,
                   plot_type="body_graph", x_axis="frequency", lags=0, frequencies=None, average="subject",
                   normalization=None, show_average=False, show_sd=False, error_type="standard_deviation",
                   show_peaks=False, show_peaks_sd=False, show=True, path_png=None, **kwargs):
    results_list = _load_coherence_files(kinematic_measure, coherence_with, level, re_referenced, average)
    _, first_results = results_list[0]
    title = _print_results_summary(results_list, kinematic_measure, coherence_with)

    if frequencies is not None:
        closest_freqs = _get_closest_frequencies(first_results, frequencies)

    if x_axis == "frequency":
        closest_lags = _get_closest_lags(first_results, lags)
        if plot_type == "body_graph":
            _plot_body_graph(results_list, kind, kinematic_measure, closest_lags,
                             normalization, show_average, title, show, path_png, **kwargs)
        elif plot_type == "silhouette":
            if frequencies is None:
                raise ValueError("'frequencies' must be specified for plot_type='silhouette'.")
            _plot_silhouette(results_list, kind, kinematic_measure, closest_lags,
                             closest_freqs, title, show, path_png, **kwargs)

    elif x_axis == "lags":
        if frequencies is None:
            raise ValueError("'frequencies' must be specified when x_axis='lags'.")
        if normalization is not None:
            print("  Warning: normalization is not applied when x_axis='lags'.")
        _plot_lags(results_list, kind, kinematic_measure, closest_freqs,
                   show_average, title, show, path_png,
                   show_sd=show_sd, error_type=error_type,
                   show_peaks=show_peaks, show_peaks_sd=show_peaks_sd, **kwargs)


if __name__ == "__main__":
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "font.family": "Metropolis",
                     "xtick.labelsize": 8, "ytick.labelsize": 8}
    kinematic_measure = "velocity"
    acoustic_measure = "envelope"
    average = "subject"
    re_referenced = True
    trailing_file_name = ""
    if re_referenced:
        trailing_file_name = REREF_SUFFIX

    # Coherence
    plot_coherence(kinematic_measure, acoustic_measure, kind="average",
                   level=ALL_SUBJECTS_MINUS_S004,
                   re_referenced=re_referenced, normalization=None, show_average="below", average=average,
                   xlim=[0, 4], lags=0, show=True, legend=False,
                   joint_layout="Qualisys",
                   sns_kwargs={"style": "whitegrid", "rc": custom_params},
                   mpl_kwargs={"hspace": 0.5, "tick_pad": -3, "xlabel_pad": 1, "ylabel_pad": 1,
                               "title_pad": 3, "title_weight": "bold", "xtick_step": 1, "tight_layout": False},
                   x_label = "Frequency (Hz)", y_label = "Coherence",
                   show_peaks=False, show_peaks_sd=False, error_type="standard_error",
                   path_png = op.join(PATH_PLOTS, "Coherence", f"coherence_{kinematic_measure}_{acoustic_measure}{trailing_file_name}.png"))

    plt.close()

    # Lags
    plot_coherence(kinematic_measure, acoustic_measure, kind="average", x_axis="lags",
                   level=ALL_SUBJECTS_MINUS_S004,
                   re_referenced=re_referenced, normalization=None, show_average="below", average=average,
                   lags=0, show=True, legend=False, frequencies=[1], xlim = [-1000, 1000],
                   joint_layout="Qualisys",
                   sns_kwargs={"style": "whitegrid", "rc": custom_params},
                   mpl_kwargs={"hspace": 0.5, "tick_pad": -3, "xlabel_pad": 1, "ylabel_pad": 1,
                               "title_pad": 3, "title_weight": "bold", "xtick_step": 1000, "tight_layout": False},
                   x_label = "Frequency (Hz)", y_label = "Coherence",
                   show_peaks=True, show_peaks_sd=True, error_type="standard_error",
                   path_png = op.join(PATH_PLOTS, "Coherence", f"coherence_{kinematic_measure}_{acoustic_measure}_lags{trailing_file_name}.png"))


