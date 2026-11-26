import math
import os
import os.path as op
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from constants import *
from helper_functions import *

PATH_FIGURES = op.join(PATH_PLOTS, "Cross-correlation", "Audio")
os.makedirs(PATH_FIGURES, exist_ok=True)


# ==========================================================================================================
# Statistics
# ==========================================================================================================

def _peak_idx_in_range(curve: np.ndarray, lags_ms: np.ndarray, peak_search_range_ms=None) -> int:
    """Index (into the full curve/lags_ms arrays) of the maximum, optionally restricted to a lag window.

    peak_search_range_ms: None (search the full lag range, the original behaviour) — a single number R
        (restricts the search to [-R, +R]) — or a (min_ms, max_ms) tuple for an asymmetric window."""
    if peak_search_range_ms is None:
        return int(np.argmax(curve))
    if isinstance(peak_search_range_ms, (tuple, list)):
        lo, hi = peak_search_range_ms
    else:
        lo, hi = -peak_search_range_ms, peak_search_range_ms
    mask = (lags_ms >= lo) & (lags_ms <= hi)
    if not mask.any():
        raise ValueError(f"No lags found within peak_search_range_ms={peak_search_range_ms}.")
    return int(np.flatnonzero(mask)[np.argmax(curve[mask])])


def test_r_at_lag0(per_subject_curves: list, lags_ms: np.ndarray, label: str, verbose: bool = True
                   ) -> tuple[float, float]:
    """One-sample Wilcoxon signed-rank test: is r at lag=0 significantly different from 0?
    verbose=False silences the print (used when only the p-value is needed, e.g. to pool across
    measures for print_corrected_pvalues).
    Returns (W statistic, p-value)."""
    lag0_idx = int(np.argmin(np.abs(lags_ms)))
    r_values = np.array([rs[lag0_idx] for rs in per_subject_curves])
    mean_r = float(np.mean(r_values))
    try:
        stat, p = wilcoxon(r_values)
    except ValueError:
        stat, p = np.nan, 1.0
    if verbose:
        tick = "✅" if p < 0.05 else "❌"
        sig_str = "significant" if p < 0.05 else "not significant"
        subj_rs_str = ", ".join([f"{r:.4f}" for r in r_values])
        print(f"  {tick} [{label}] r at lag=0: mean r = {mean_r:.4f}, Wilcoxon W = {stat:.1f}, p = {p:.4f} "
             f"({sig_str})")
        print(f"       Subject r at lag=0: [{subj_rs_str}]")
    return float(stat), float(p)


def test_peak_lag_vs_0(per_subject_curves: list, lags_ms: np.ndarray, label: str,
                       peak_search_range_ms=None, verbose: bool = True) -> tuple[float, float]:
    """One-sample Wilcoxon signed-rank test: is the peak lag significantly different from 0?
    peak_search_range_ms: see _peak_idx_in_range — restricts where the per-curve peak is looked for,
    e.g. to keep noisy single-joint/single-subject curves from picking up a spurious peak far out in
    the tails of the lag range.
    verbose=False silences the print (used when only the p-value is needed, e.g. to pool across
    measures for print_corrected_pvalues).
    Returns (W statistic, p-value)."""
    peak_idxs = [_peak_idx_in_range(rs, lags_ms, peak_search_range_ms) for rs in per_subject_curves]
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
    if verbose:
        tick = "✅" if p < 0.05 else "❌"
        sig_str = "significant" if p < 0.05 else "not significant"
        subj_lags_str = ", ".join([f"{float(l):.0f}" for l in peak_lags])
        subj_rs_str = ", ".join([f"{r:.4f}" for r in peak_rs])
        print(f"  {tick} [{label}] peak lag vs 0: mean = {mean_lag:.1f} ms (SD = {sd_lag:.1f}, SE = {se_lag:.1f}), "
             f"mean peak r = {mean_r:.4f}, Wilcoxon W = {stat:.1f}, p = {p:.4f} ({sig_str})")
        print(f"       Subject peak lags: [{subj_lags_str}] ms")
        print(f"       Subject peak r:    [{subj_rs_str}]")
    return float(stat), float(p)


def run_stats_on_xcorr(xcorr_per_joint: dict, xcorr_per_group: dict, lags_ms: np.ndarray, joints=None,
                       groups=None, peak_search_range_ms=None) -> dict:
    """Run both Wilcoxon tests (r at lag=0, peak lag vs 0) for a set of joints and/or groups of a
    loaded cross-correlation result. `joints`/`groups` accept a single name, a list, or None (= every
    joint/group present). peak_search_range_ms: see _peak_idx_in_range, forwarded to the peak-lag test.
    Returns {label: {"lag0": (W, p), "peak_lag": (W, p)}}."""
    if joints is None:
        joints = list(xcorr_per_joint.keys())
    elif isinstance(joints, str):
        joints = [joints]
    if groups is None:
        groups = list(xcorr_per_group.keys())
    elif isinstance(groups, str):
        groups = [groups]

    results = {}
    for joint in joints:
        curves = xcorr_per_joint.get(joint)
        if not curves:
            continue
        stat0, p0 = test_r_at_lag0(curves, lags_ms, joint)
        statp, pp = test_peak_lag_vs_0(curves, lags_ms, joint, peak_search_range_ms=peak_search_range_ms)
        results[joint] = {"lag0": (stat0, p0), "peak_lag": (statp, pp)}

    for group in groups:
        curves = xcorr_per_group.get(group)
        if not curves:
            continue
        stat0, p0 = test_r_at_lag0(curves, lags_ms, group)
        statp, pp = test_peak_lag_vs_0(curves, lags_ms, group, peak_search_range_ms=peak_search_range_ms)
        results[group] = {"lag0": (stat0, p0), "peak_lag": (statp, pp)}

    return results


def print_peak_lag_table(n, kinematic_measure: str, acoustic_measure: str, groups=None, all_subjects=None,
                         subjects=None, exclude_subjects=None, fs: float = None, re_referenced: bool = True,
                         file_trail: str = "", peak_search_range_ms=None) -> dict:
    """Print a tab-separated table (one row per subject, one column per group) of each subject's peak
    lag in ms — paste straight into Excel. Below the subject rows, print each group's Wilcoxon peak-lag-
    vs-0 p-value and whether it's significant (uncorrected).

    groups: defaults to ["Average of all joints", "HandRight", "HandLeft", "Head"].
    peak_search_range_ms: see _peak_idx_in_range — restricts where each subject's peak is looked for
        (None searches the full lag range, the original behaviour).
    Returns {group: p_value} (uncorrected) so callers can pool p-values across measures for correction —
    see print_corrected_pvalues."""
    groups = groups or ["Average of all joints", "HandRight", "HandLeft", "Head"]

    all_subjects = all_subjects or ALL_SUBJECTS
    keep_idx = _resolve_subject_indices(all_subjects, subjects, exclude_subjects)
    kept_subjects = [all_subjects[i] for i in keep_idx]
    subject_labels = [s.split("_", 1)[-1] if "_" in s else s for s in kept_subjects]

    _, _, _, xcorr_per_group, lags_ms = load_xcorr_results(kinematic_measure, acoustic_measure, fs=fs,
                                                            re_referenced=re_referenced, file_trail=file_trail)

    title = f"{n}. PEAK LAG TABLE (ms) — {kinematic_measure.upper()} x {acoustic_measure.upper()}"
    if peak_search_range_ms is not None:
        title += f" (search range: ±{peak_search_range_ms} ms)" if not isinstance(peak_search_range_ms,
                 (tuple, list)) else f" (search range: {peak_search_range_ms} ms)"
    print_section_title(title)

    peak_lags_by_group = {}
    p_by_group = {}
    min_n = len(keep_idx)
    for group in groups:
        curves = xcorr_per_group.get(group)
        if not curves:
            print(f"  [!] No data for group '{group}', skipping.")
            continue
        curves_kept = [curves[i] for i in keep_idx if i < len(curves)]
        min_n = min(min_n, len(curves_kept))
        peak_idxs = [_peak_idx_in_range(c, lags_ms, peak_search_range_ms) for c in curves_kept]
        peak_lags_by_group[group] = np.array([lags_ms[idx] for idx in peak_idxs])
        try:
            _, p = wilcoxon(peak_lags_by_group[group])
        except ValueError:
            p = 1.0
        p_by_group[group] = float(p)

    if min_n < len(keep_idx):
        print(f"  [!] Expected {len(keep_idx)} subjects but the shortest group has only {min_n} saved curves "
             f"for {acoustic_measure} — some subjects in all_subjects were likely added after this pickle was "
             f"last generated by B01. Showing only the first {min_n} subjects (by all_subjects order) that "
             f"every group actually has data for; re-run B01 for this measure to include the rest.")
        subject_labels = subject_labels[:min_n]
        peak_lags_by_group = {g: v[:min_n] for g, v in peak_lags_by_group.items()}

    available_groups = list(peak_lags_by_group.keys())
    print("Subject\t" + "\t".join(available_groups))
    for i, subj_label in enumerate(subject_labels):
        row = [subj_label] + [f"{peak_lags_by_group[g][i]:.0f}" for g in available_groups]
        print("\t".join(row))
    print("p\t" + "\t".join(f"{p_by_group[g]:.4f}" for g in available_groups))
    print("significant\t" + "\t".join(("yes" if p_by_group[g] < 0.05 else "no") for g in available_groups))

    return p_by_group


def print_corrected_pvalues(n, pvals_by_group_measure: dict, acoustic_measures: list,
                            method: str = "fdr_bh", test_label: str = "peak lag vs 0") -> None:
    """Benjamini-Hochberg-correct (matching the convention used for the correlation matrix in
    C03_acoustic_analyses.py, which is the closer analogy — no omnibus test gates this family, it's
    just the same test run once per acoustic measure) the p-values collected across acoustic measures —
    one family of `len(acoustic_measures)` comparisons per group (kinematic measure held fixed to
    velocity, so only the acoustic measure varies within each family). Prints two tab-separated tables
    (corrected p, then significance) ready to paste into Excel.

    pvals_by_group_measure: {group: {acoustic_measure: raw_p}}, e.g. the dicts returned by
    print_peak_lag_table / collect_group_pvalues accumulated across a loop over acoustic_measures.
    test_label: only used for the section title, e.g. "peak lag vs 0" or "r at lag=0"."""
    print_section_title(f"{n}. CORRECTED P-VALUES ({method}, {test_label}) — velocity x "
                        f"{{{', '.join(acoustic_measures)}}} ({len(acoustic_measures)} comparisons)")

    groups = [g for g in pvals_by_group_measure if all(m in pvals_by_group_measure[g] for m in acoustic_measures)]
    skipped = [g for g in pvals_by_group_measure if g not in groups]
    if skipped:
        print(f"  [!] Skipping group(s) with incomplete data across the {len(acoustic_measures)} measures: "
             f"{', '.join(skipped)}")
    if not groups:
        print("  [!] No group has complete data across all acoustic measures — nothing to correct.")
        return

    corrected = {}
    for group in groups:
        raw_p = [pvals_by_group_measure[group][m] for m in acoustic_measures]
        _, p_corr, _, _ = multipletests(raw_p, method=method)
        corrected[group] = dict(zip(acoustic_measures, p_corr))

    print("Acoustic measure\t" + "\t".join(groups))
    for m in acoustic_measures:
        print("\t".join([m] + [f"{corrected[g][m]:.4f}" for g in groups]))

    print()
    print("Acoustic measure\t" + "\t".join(groups))
    for m in acoustic_measures:
        print("\t".join([m] + [("yes" if corrected[g][m] < 0.05 else "no") for g in groups]))


def collect_group_pvalues(kinematic_measure: str, acoustic_measure: str, groups, test: str = "peak_lag",
                          all_subjects=None, subjects=None, exclude_subjects=None, fs: float = None,
                          re_referenced: bool = True, file_trail: str = "", peak_search_range_ms=None) -> dict:
    """Silently (no console output) compute one Wilcoxon p-value per group for one kinematic × acoustic
    measure pair — for pooling across acoustic measures before handing to print_corrected_pvalues.

    test: "lag0" (r at lag=0) or "peak_lag" (peak lag vs 0, honours peak_search_range_ms).
    Returns {group: p_value}."""
    if test not in ("lag0", "peak_lag"):
        raise ValueError(f"test must be 'lag0' or 'peak_lag', got {test!r}")

    all_subjects = all_subjects or ALL_SUBJECTS
    keep_idx = _resolve_subject_indices(all_subjects, subjects, exclude_subjects)

    _, _, _, xcorr_per_group, lags_ms = load_xcorr_results(kinematic_measure, acoustic_measure, fs=fs,
                                                            re_referenced=re_referenced, file_trail=file_trail)

    p_by_group = {}
    for group in groups:
        curves = xcorr_per_group.get(group)
        if not curves:
            continue
        curves_kept = [curves[i] for i in keep_idx if i < len(curves)]
        if test == "lag0":
            _, p = test_r_at_lag0(curves_kept, lags_ms, group, verbose=False)
        else:
            _, p = test_peak_lag_vs_0(curves_kept, lags_ms, group, peak_search_range_ms=peak_search_range_ms,
                                      verbose=False)
        p_by_group[group] = p

    return p_by_group


def compute_group_stats(kinematic_measure: str, acoustic_measure: str, groups, all_subjects=None, subjects=None,
                        exclude_subjects=None, fs: float = None, re_referenced: bool = True, file_trail: str = "",
                        peak_search_range_ms=None) -> dict:
    """Silently (no console output) compute the full r-at-lag0 and peak-lag stats — not just p, unlike
    collect_group_pvalues — for one kinematic × acoustic measure pair, for every group in `groups`.

    Returns {group: {"lag0": {"r": mean_r, "p": p}, "peak_lag": {"l": mean_lag, "sd": sd_lag, "r": mean_r,
    "p": p}}}. `l` here is the raw value in the same sign convention as the console prints (positive =
    kinematics compared against later acoustic samples) — see print_summary_table's flip_lag_sign for
    converting to the write-up's convention (negative = kinematics precedes acoustics)."""
    all_subjects = all_subjects or ALL_SUBJECTS
    keep_idx = _resolve_subject_indices(all_subjects, subjects, exclude_subjects)

    _, _, _, xcorr_per_group, lags_ms = load_xcorr_results(kinematic_measure, acoustic_measure, fs=fs,
                                                            re_referenced=re_referenced, file_trail=file_trail)
    lag0_idx = int(np.argmin(np.abs(lags_ms)))

    stats = {}
    for group in groups:
        curves = xcorr_per_group.get(group)
        if not curves:
            continue
        curves_kept = [curves[i] for i in keep_idx if i < len(curves)]

        r0_values = np.array([c[lag0_idx] for c in curves_kept])
        try:
            _, p0 = wilcoxon(r0_values)
        except ValueError:
            p0 = 1.0

        peak_idxs = [_peak_idx_in_range(c, lags_ms, peak_search_range_ms) for c in curves_kept]
        peak_lags = np.array([lags_ms[i] for i in peak_idxs])
        peak_rs = np.array([float(curves_kept[s][i]) for s, i in enumerate(peak_idxs)])
        try:
            _, p_peak = wilcoxon(peak_lags)
        except ValueError:
            p_peak = 1.0

        stats[group] = {
            "lag0": {"r": float(r0_values.mean()), "p": float(p0)},
            "peak_lag": {"l": float(peak_lags.mean()), "sd": float(peak_lags.std()),
                        "r": float(peak_rs.mean()), "p": float(p_peak)},
        }

    return stats


def print_summary_table(n, kinematic_measure: str, acoustic_measures: list, groups=None, group_labels=None,
                        flip_lag_sign: bool = True, corrected: bool = False, correction_method: str = "fdr_bh",
                        all_subjects=None, subjects=None, exclude_subjects=None, fs: float = None,
                        re_referenced: bool = True, file_trail: str = "", peak_search_range_ms=None) -> None:
    """Print a tab-separated summary table (r/p at lag=0, then l/sd/r/p for peak lag — one block of rows
    per acoustic measure, one column per group) ready to paste straight into Excel.

    groups: defaults to ["Average of all joints", "HandRight", "HandLeft", "Head"] — pass e.g.
        [..., "HeadFront"] for datasets (like Qualisys here) that kept the original head group name.
    group_labels: {internal_group_name: column_header}, defaults to the obvious short labels
        ("Average of all joints" -> "Average", "HandRight" -> "Right hand", etc.) — "HeadFront" maps to
        "Head" too, so the Qualisys naming doesn't need a separate label dict.
    flip_lag_sign: negates every peak-lag `l` value before printing, matching the write-up convention
        (negative = kinematics precedes acoustics) rather than the script's raw sign (positive =
        kinematics compared against later acoustic samples). Set False to print the raw, unflipped
        values instead.
    corrected: False (default) prints the raw p from a single Wilcoxon test per cell, same as sections
        1-10. True BH-corrects — matching print_corrected_pvalues's convention — each group's p-values
        as their own family of len(acoustic_measures) comparisons, applied separately to the lag0 p's
        and the peak_lag p's (so a group's lag0 family and its peak_lag family are corrected
        independently, not pooled together). The section title states which mode was used."""
    groups = groups or ["Average of all joints", "HandRight", "HandLeft", "Head"]
    default_labels = {"Average of all joints": "Average", "HandRight": "Right hand", "HandLeft": "Left hand",
                      "Head": "Head", "HeadFront": "Head"}
    group_labels = {**default_labels, **(group_labels or {})}
    labels = [group_labels.get(g, g) for g in groups]

    mode_label = f"{correction_method}-corrected" if corrected else "uncorrected"
    print_section_title(f"{n}. SUMMARY TABLE ({mode_label}) — {kinematic_measure.upper()} x "
                        f"{{{', '.join(acoustic_measures)}}}")

    all_stats = {}
    for acoustic_measure in acoustic_measures:
        all_stats[acoustic_measure] = compute_group_stats(kinematic_measure, acoustic_measure, groups,
                                                           all_subjects=all_subjects, subjects=subjects,
                                                           exclude_subjects=exclude_subjects, fs=fs,
                                                           re_referenced=re_referenced, file_trail=file_trail,
                                                           peak_search_range_ms=peak_search_range_ms)

    if corrected:
        for group in groups:
            measures_with_group = [m for m in acoustic_measures if group in all_stats[m]]
            if not measures_with_group:
                continue
            lag0_raw = [all_stats[m][group]["lag0"]["p"] for m in measures_with_group]
            peak_raw = [all_stats[m][group]["peak_lag"]["p"] for m in measures_with_group]
            _, lag0_corr, _, _ = multipletests(lag0_raw, method=correction_method)
            _, peak_corr, _, _ = multipletests(peak_raw, method=correction_method)
            for m, p0, pp in zip(measures_with_group, lag0_corr, peak_corr):
                all_stats[m][group]["lag0"]["p"] = float(p0)
                all_stats[m][group]["peak_lag"]["p"] = float(pp)

    def cell(measure, group, section, key, fmt):
        s = all_stats[measure].get(group)
        if s is None:
            return "?"
        v = s[section][key]
        if section == "peak_lag" and key == "l" and flip_lag_sign:
            v = -v
        return f"{v:{fmt}}"

    print("\t\tLag 0")
    print("\t\t" + "\t".join(labels))
    for measure in acoustic_measures:
        title = measure.capitalize()
        print(f"{title}\tr\t" + "\t".join(cell(measure, g, "lag0", "r", ".4f") for g in groups))
        print("\tp\t" + "\t".join(cell(measure, g, "lag0", "p", ".4f") for g in groups))

    print()
    print()
    print("\t\tPeak lag")
    print("\t\t" + "\t".join(labels))
    for measure in acoustic_measures:
        title = measure.capitalize()
        print(f"{title}\tl\t" + "\t".join(cell(measure, g, "peak_lag", "l", ".1f") for g in groups))
        print("\tsd\t" + "\t".join(cell(measure, g, "peak_lag", "sd", ".1f") for g in groups))
        print("\tr\t" + "\t".join(cell(measure, g, "peak_lag", "r", ".4f") for g in groups))
        print("\tp\t" + "\t".join(cell(measure, g, "peak_lag", "p", ".4f") for g in groups))


# ==========================================================================================================
# Loading + filtering previously computed results (saved by B01_cross_correlation.py)
# ==========================================================================================================

# B01's own joint_groups dict (in its __main__ block) keys the right hand as "HandInRight" instead of
# "HandRight" (inconsistent with "HandLeft": ["HandInLeft"] for the left hand) — likely a copy-paste slip.
# That name is baked into every pickle B01 already wrote, so load_xcorr_results relabels it on load rather
# than requiring B01 to be re-run. Fix it at the source in B01 and set this to {} once the pickles are
# regenerated with the correct group name.
DEFAULT_GROUP_RENAME = {"HandInRight": "HandRight"}


def load_xcorr_results(kinematic_measure: str, acoustic_measure: str, fs: float = None, re_referenced: bool = True,
                       file_trail: str = "", group_rename: dict = None) -> tuple[dict, dict, dict, dict, np.ndarray]:
    """Load the group-level cross-correlation results pickled by B01 for one kinematic × acoustic
    measure pair. Returns (xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group, lags_ms),
    matching what B01 dumps to disk.

    group_rename: {old_group_name: new_group_name}, applied to xcorr_group_mean / xcorr_per_group keys
        after loading — a cosmetic relabel only, no recomputation. Defaults to DEFAULT_GROUP_RENAME
        (fixes B01's "HandInRight" group label); pass {} to disable."""
    fs = fs or FS
    group_rename = DEFAULT_GROUP_RENAME if group_rename is None else group_rename
    trailing_file_name = (REREF_SUFFIX if re_referenced else "") + file_trail
    path = op.join(PATH_RESULTS, "Cross-correlation", "Audio", "Group",
                   f"xcorr_{SYSTEM.lower()}_fs{int(fs)}_{kinematic_measure}_{acoustic_measure}"
                   f"{trailing_file_name}.pkl")
    with open(path, "rb") as f:
        xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group, lags_ms = pickle.load(f)

    for old_name, new_name in group_rename.items():
        if old_name in xcorr_group_mean:
            xcorr_group_mean[new_name] = xcorr_group_mean.pop(old_name)
        if old_name in xcorr_per_group:
            xcorr_per_group[new_name] = xcorr_per_group.pop(old_name)

    return xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group, lags_ms


def _resolve_subject_indices(all_subjects: list, subjects=None, exclude_subjects=None) -> list:
    """Shared by _filter_xcorr_curves and print_peak_lag_table so both index into the same curve
    lists (positional, in `all_subjects` order) the same way."""
    keep_idx = list(range(len(all_subjects)))
    if subjects is not None:
        subj_list = [subjects] if isinstance(subjects, str) else list(subjects)
        keep_idx = [i for i in keep_idx if all_subjects[i] in subj_list]
    if exclude_subjects is not None:
        excl_list = [exclude_subjects] if isinstance(exclude_subjects, str) else list(exclude_subjects)
        keep_idx = [i for i in keep_idx if all_subjects[i] not in excl_list]
    return keep_idx


def _filter_xcorr_curves(xcorr_per_joint: dict, xcorr_per_group: dict, all_subjects: list, subjects=None,
                         exclude_subjects=None) -> tuple[dict, dict, dict, dict]:
    """The curve lists in xcorr_per_joint / xcorr_per_group are indexed in the same subject order
    B01 iterated over (`all_subjects`, defaulting to ALL_SUBJECTS — pass the exact list/order
    used when the pickle was generated if it differs). Keeps only the requested subjects and recomputes
    the joint/group means. Returns (joint_mean, group_mean, per_joint, per_group)."""
    keep_idx = _resolve_subject_indices(all_subjects, subjects, exclude_subjects)

    per_joint = {j: [curves[i] for i in keep_idx if i < len(curves)] for j, curves in xcorr_per_joint.items()}
    per_group = {g: [curves[i] for i in keep_idx if i < len(curves)] for g, curves in xcorr_per_group.items()}
    joint_mean = {j: np.nanmean(np.vstack(v), axis=0) for j, v in per_joint.items() if v}
    group_mean = {g: np.nanmean(np.vstack(v), axis=0) for g, v in per_group.items() if v}
    return joint_mean, group_mean, per_joint, per_group


# ==========================================================================================================
# Plotting — low level (unchanged from B01)
# ==========================================================================================================

def _draw_subpanel(ax: plt.Axes, subpanel: str, xcorr_joint_mean: dict, xcorr_group_mean: dict, xcorr_per_joint: dict,
                   xcorr_per_group: dict, lags_ms: np.ndarray, curve_labels=None, peak_search_range_ms=None) -> None:
    """Draw a single xcorr subpanel on ax.

    subpanel = "per_joint"   → group means in grey tones (Head: light → AllJoints: black)
    subpanel = joint name    → individual curves for that joint + their mean
    subpanel = group name    → individual curves for that group + their mean

    curve_labels: labels for the individual curves in the non-"per_joint" subpanels.
        None      → group figure: one curve per subject, coloured/labelled via SUBJECT_COLORS / SUBJECT_ANON.
        list[str] → per-subject figure: one curve per trial, labelled with these and coloured from a cycle.
    peak_search_range_ms: see _peak_idx_in_range — restricts where each curve's peak marker is placed
        (None searches the full lag range, the original behaviour).
    """
    if subpanel == "per_joint":
        # Group means in grey tones with SE bar at peak — no individual joint background
        for group, rs_mean in xcorr_group_mean.items():
            color = PER_JOINT_GREYS.get(group, "0.50")
            ax.plot(lags_ms, rs_mean, color=color, linewidth=2.2, label=group, zorder=4)
            if group in xcorr_per_group and len(xcorr_per_group[group]) > 1:
                peak_lags = np.array([lags_ms[_peak_idx_in_range(s, lags_ms, peak_search_range_ms)]
                                     for s in xcorr_per_group[group]])
                se = float(np.std(peak_lags) / np.sqrt(len(peak_lags)))
                peak_idx = _peak_idx_in_range(rs_mean, lags_ms, peak_search_range_ms)
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
            peak_idx = _peak_idx_in_range(curve, lags_ms, peak_search_range_ms)
            ax.plot(lags_ms[peak_idx], curve[peak_idx], "o", color=color, markersize=4, zorder=4)

        # Mean curve + SE bar at peak
        color_mean = PER_JOINT_GREYS.get(subpanel, "black")
        ax.plot(lags_ms, rs_mean, color=color_mean, linewidth=2.2, label="Mean", zorder=5)
        peak_lags = np.array([lags_ms[_peak_idx_in_range(curve, lags_ms, peak_search_range_ms)]
                             for curve in individual_curves])
        se = float(np.std(peak_lags) / np.sqrt(len(peak_lags)))
        peak_idx = _peak_idx_in_range(rs_mean, lags_ms, peak_search_range_ms)
        ax.errorbar(lags_ms[peak_idx], rs_mean[peak_idx], xerr=se, fmt="none", color=color_mean, capsize=4,
                    linewidth=1.5, zorder=6)

        across = "trials" if curve_labels is not None else "subjects"
        ax.set_title(f"{subpanel} across {across}", loc="center", fontsize=12, fontweight="bold")


def plot_xcorr(xcorr_joint_mean: dict, xcorr_group_mean: dict, xcorr_per_joint: dict, xcorr_per_group: dict,
               lags_ms: np.ndarray, subpanels: list, kinematic_measure: str, acoustic_measure: str, out_path: str,
               show: bool, curve_labels=None, peak_search_range_ms=None) -> None:
    """Stack subpanels vertically and save the figure. `out_path=None` skips saving.
    peak_search_range_ms: forwarded to _draw_subpanel — see _peak_idx_in_range."""
    n = len(subpanels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, subpanel in zip(axes, subpanels):
        _draw_subpanel(ax, subpanel, xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group, lags_ms,
                       curve_labels=curve_labels, peak_search_range_ms=peak_search_range_ms)
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
    if out_path is not None:
        os.makedirs(op.split(out_path)[0], exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ==========================================================================================================
# Flexible display functions (mirrors the style of C03_acoustic_analyses.py: numbered sections, list-or-str
# measure args, optional subject filtering, an `ax` hook for building your own grids, show/save toggles)
# ==========================================================================================================

def display_xcorr(n, kinematic_measure: str, acoustic_measure: str, subpanels=None, subjects=None,
                  exclude_subjects=None, all_subjects=None, fs: float = None, re_referenced: bool = True,
                  file_trail: str = "", show_stats: bool = True, show: bool = True, path_png: str = None, ax=None,
                  title: str = None, peak_search_range_ms=None) -> dict:
    """Load one kinematic × acoustic cross-correlation result and display it.

    subpanels: "per_joint" / a joint name / a group name, or a list of those. Defaults to
        ["per_joint", "Average of all joints"].
    subjects / exclude_subjects: restrict the displayed (and tested) curves to a subset of subjects,
        recomputing the means on the fly — no need to regenerate the pickle from B01.
    ax: draw a single subpanel onto an existing axis instead of building a new figure (for use in
        display_xcorr_multi_panel or your own custom grids) — requires exactly one subpanel.
    peak_search_range_ms: see _peak_idx_in_range — restricts where peak lags are looked for, both in the
        stats (test_peak_lag_vs_0) and in the peak markers drawn on the curves. None (default) searches
        the full lag range, unchanged from before.
    Returns the dict of Wilcoxon test results (empty if show_stats=False).
    """
    print_section_title(f"{n}. CROSS-CORRELATION — {kinematic_measure.upper()} x {acoustic_measure.upper()}")

    if subpanels is None:
        subpanels = ["per_joint", "Average of all joints"]
    elif isinstance(subpanels, str):
        subpanels = [subpanels]

    if ax is not None and len(subpanels) != 1:
        raise ValueError("When drawing onto an external `ax`, `subpanels` must contain exactly one "
                         "subpanel (one axis = one subpanel) — for several subpanels, omit `ax` and "
                         "let the function build its own stacked figure.")

    all_subjects = all_subjects or ALL_SUBJECTS

    _, _, xcorr_per_joint, xcorr_per_group, lags_ms = load_xcorr_results(kinematic_measure, acoustic_measure, fs=fs,
        re_referenced=re_referenced, file_trail=file_trail)
    xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group = _filter_xcorr_curves(xcorr_per_joint,
        xcorr_per_group, all_subjects, subjects, exclude_subjects)

    stats_results = {}
    if show_stats:
        stats_targets = [s for s in subpanels if s != "per_joint"]
        stats_results = run_stats_on_xcorr(xcorr_per_joint, xcorr_per_group, lags_ms, joints=stats_targets,
                                           groups=stats_targets, peak_search_range_ms=peak_search_range_ms)

    if ax is not None:
        _draw_subpanel(ax, subpanels[0], xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group,
                      lags_ms, peak_search_range_ms=peak_search_range_ms)
        ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Pearson r")
        ax.set_xlabel("Lag (ms)")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.set_xlim(lags_ms[0], lags_ms[-1])
        ax.invert_xaxis()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(-x)}"))
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        return stats_results

    plot_xcorr(xcorr_joint_mean, xcorr_group_mean, xcorr_per_joint, xcorr_per_group, lags_ms, subpanels,
               kinematic_measure, acoustic_measure, path_png, show=show, peak_search_range_ms=peak_search_range_ms)
    return stats_results


def display_xcorr_multi_panel(n, kinematic_measures, acoustic_measures, subpanel: str = "Average of all joints",
                              ncols: int = None, figsize=None, subjects=None, exclude_subjects=None, all_subjects=None,
                              fs: float = None, re_referenced: bool = True, file_trail: str = "",
                              show_stats: bool = True, show: bool = True, path_png: str = None,
                              peak_search_range_ms=None) -> None:
    """Grid of kinematic × acoustic combos, one subpanel per cell — mirrors plot_acoustic_multi_panel
    in C03. `kinematic_measures` / `acoustic_measures` each accept a single measure or a list; every
    combination of the two is plotted. peak_search_range_ms: forwarded to display_xcorr for each cell."""
    kinematic_measures = [kinematic_measures] if isinstance(kinematic_measures, str) else list(kinematic_measures)
    acoustic_measures = [acoustic_measures] if isinstance(acoustic_measures, str) else list(acoustic_measures)
    combos = [(k, a) for k in kinematic_measures for a in acoustic_measures]

    print_section_title(f"{n}. CROSS-CORRELATION — {subpanel.upper()} (multi-panel)")

    ncols = ncols or len(combos)
    nrows = math.ceil(len(combos) / ncols)
    figsize = figsize or (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, (kinematic_measure, acoustic_measure) in enumerate(combos):
        display_xcorr(n, kinematic_measure, acoustic_measure, subpanels=subpanel, subjects=subjects,
                      exclude_subjects=exclude_subjects, all_subjects=all_subjects, fs=fs, re_referenced=re_referenced,
                      file_trail=file_trail, show_stats=show_stats, show=False, ax=axes_flat[idx],
                      title=f"{kinematic_measure} × {acoustic_measure}", peak_search_range_ms=peak_search_range_ms)

    for idx in range(len(combos), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout()
    if path_png is not None:
        os.makedirs(op.split(path_png)[0], exist_ok=True)
        plt.savefig(path_png, dpi=150)
        print(f"Saved → {path_png}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    display_figures = False
    kinematic_measure = "velocity"
    peak_lag_groups = ["Average of all joints", "HandRight", "HandLeft", "Head"]
    peak_search_range_ms = 1000  # restricts peak-lag search to ±1000 ms; set to None for the full ±10000 ms range
    i = 1

    pvals_peak_lag = {g: {} for g in peak_lag_groups}
    pvals_lag0 = {g: {} for g in peak_lag_groups}

    for acoustic_measure in DERIVATIVES:
        display_xcorr(i, kinematic_measure, acoustic_measure, subpanels=["per_joint", "Average of all joints"],
                     show_stats=True, show=display_figures, path_png=op.join(PATH_FIGURES, "Group",
                     f"xcorr_{SYSTEM.lower()}_fs{int(FS)}_{kinematic_measure}_{acoustic_measure}{REREF_SUFFIX}.png"),
                     peak_search_range_ms=peak_search_range_ms)
        i += 1

        p_by_group = print_peak_lag_table(i, kinematic_measure, acoustic_measure, groups=peak_lag_groups,
                                          peak_search_range_ms=peak_search_range_ms)
        for group, p in p_by_group.items():
            pvals_peak_lag[group][acoustic_measure] = p
        i += 1

        lag0_p_by_group = collect_group_pvalues(kinematic_measure, acoustic_measure, peak_lag_groups, test="lag0")
        for group, p in lag0_p_by_group.items():
            pvals_lag0[group][acoustic_measure] = p

    print_corrected_pvalues(i, pvals_peak_lag, DERIVATIVES, test_label="peak lag vs 0")
    i += 1
    print_corrected_pvalues(i, pvals_lag0, DERIVATIVES, test_label="r at lag=0")
    i += 1

    print_summary_table(i, kinematic_measure, DERIVATIVES, groups=peak_lag_groups,
                        peak_search_range_ms=peak_search_range_ms, corrected=False)
    i += 1
    print_summary_table(i, kinematic_measure, DERIVATIVES, groups=peak_lag_groups,
                        peak_search_range_ms=peak_search_range_ms, corrected=True)

    # display_xcorr_multi_panel(2, ["velocity", "acceleration"], ["envelope", "intensity", "pitch"],
    #                          subpanel="Average of all joints", show_stats=False, show=display_figures)