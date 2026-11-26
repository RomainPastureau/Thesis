import hashlib
import json
import os

from pathlib import Path

from constants import *
from digestible_results import *

import numpy as np
import pandas as pd

from scipy.special import logit
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm
from scipy.stats import chi2

from krajjat import Results

from digestible_results import _interpolate_gradient, _rgb_text, _anova_term_for_columns, _p_colored, _p_to_stars, \
    _make_fixed_effect_terms, _as_list, print_digestible_results, print_digestible_results_corrected, \
    COMPARE_TO_COLUMN, _get_anova_effect_p, _get_lmm_effect_p_for_digest, _bh_correct, _p_digestible_colored, \
    _format_p_tabs, _merge_effect_summaries, _find_lmm_coefficient_p_for_level

init(autoreset=True, strip=False)



# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _as_compare_tuple(compare):
    """Normalize compare argument."""
    allowed = set(COMPARE_TO_COLUMN)

    if isinstance(compare, str):
        compare = (compare,)
    elif isinstance(compare, list):
        compare = tuple(compare)
    elif not isinstance(compare, tuple):
        raise TypeError(f"`compare` must be a string, list, or tuple. Allowed values are: {sorted(allowed)}")

    unknown = set(compare) - allowed
    if unknown:
        raise ValueError(f"Unknown compare level(s): {sorted(unknown)}. "
                         f"Allowed values are: {sorted(allowed)}")

    return compare


def _normalize_freq_specs(freq):
    """
    Normalize freq argument.

    Accepted:
    - 0.4
    - (0.1, 0.5)
    - [0.2, 0.4, (0.1, 0.5)]
    - None, meaning all available frequencies are kept separately.
    """
    if freq is None:
        return [{"kind": "all", "label": "all"}]

    if isinstance(freq, tuple):
        if len(freq) != 2:
            raise ValueError("A frequency tuple must be (lower_bound, upper_bound).")
        lo, hi = float(freq[0]), float(freq[1])
        if lo > hi:
            lo, hi = hi, lo
        return [{"kind": "band", "lower": lo, "upper": hi, "label": f"{lo:g}-{hi:g} Hz", }]

    if isinstance(freq, list):
        specs = []
        for f in freq:
            specs.extend(_normalize_freq_specs(f))
        return specs

    f = float(freq)
    return [{"kind": "point", "frequency": f, "label": f"{f:g} Hz", }]


def _select_frequency_values(values, frequencies, freq_specs, freq_atol=None):
    """
    Return rows containing selected/aggregated coherence values for requested frequencies.

    For point frequencies, the nearest available frequency is selected.
    For frequency bands, values are averaged within the band.
    For freq=None, all frequencies are returned separately.
    """
    values = np.asarray(values, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)

    rows = []

    for spec in freq_specs:
        if spec["kind"] == "all":
            for f, v in zip(frequencies, values):
                rows.append(
                    {"freq_label": f"{f:g} Hz", "freq_kind": "point", "frequency": float(f), "freq_lower": float(f),
                     "freq_upper": float(f), "value": float(v), })

        elif spec["kind"] == "point":
            requested = spec["frequency"]
            idx = int(np.argmin(np.abs(frequencies - requested)))
            selected = float(frequencies[idx])

            if freq_atol is not None and abs(selected - requested) > freq_atol:
                raise ValueError(f"Requested frequency {requested:g} Hz, but closest available "
                                 f"frequency is {selected:g} Hz, exceeding freq_atol={freq_atol}.")

            rows.append(
                {"freq_label": spec["label"], "freq_kind": "point", "frequency": selected, "freq_lower": selected,
                 "freq_upper": selected, "value": float(values[idx]), })

        elif spec["kind"] == "band":
            lo, hi = spec["lower"], spec["upper"]
            mask = (frequencies >= lo) & (frequencies <= hi)

            if not np.any(mask):
                raise ValueError(f"No frequency found between {lo:g} and {hi:g} Hz. "
                                 f"Available range: {frequencies.min():g}-{frequencies.max():g} Hz.")

            rows.append(
                {"freq_label": spec["label"], "freq_kind": "band", "frequency": float(np.mean(frequencies[mask])),
                 "freq_lower": lo, "freq_upper": hi, "value": float(np.nanmean(values[mask])), })

        else:
            raise ValueError(f"Unknown frequency spec kind: {spec['kind']}")

    return rows


def _find_matching_key(d, requested, *, name="key", atol=1e-9):
    """
    Find exact or numerically close key in a dict.

    Useful because lags may be stored as floats such as 0.6000000000000001.
    """
    if requested in d:
        return requested

    try:
        requested_float = float(requested)
    except (TypeError, ValueError):
        raise KeyError(f"{name} {requested!r} not found. Available keys: {list(d.keys())[:10]}...")

    numeric_keys = []
    for k in d.keys():
        try:
            numeric_keys.append((k, float(k)))
        except (TypeError, ValueError):
            pass

    if not numeric_keys:
        raise KeyError(f"{name} {requested!r} not found. Available keys: {list(d.keys())[:10]}...")

    closest_key, closest_value = min(numeric_keys, key=lambda kv: abs(kv[1] - requested_float))

    if abs(closest_value - requested_float) <= atol:
        return closest_key

    raise KeyError(f"{name} {requested!r} not found. Closest available value is {closest_value!r}.")


def _find_target_measure_key(results_dict, acoustic_measure):
    """
    Find the target-measure key corresponding to one acoustic measure.

    Expected key is usually ('Audio', acoustic_measure), but this is more tolerant.
    """
    expected = ("Audio", acoustic_measure)

    if expected in results_dict:
        return expected

    for key in results_dict.keys():
        if isinstance(key, tuple) and len(key) >= 2 and key[1] == acoustic_measure:
            return key

    raise KeyError(f"Could not find target measure for acoustic_measure={acoustic_measure!r}. "
                   f"Available keys: {list(results_dict.keys())}")


def _make_cache_path(subject, freq, joints, kinematic_measure, acoustic_measure, lag, average, value_type,
                     condition=None, use_condition_results=False, ):
    """Create a deterministic cache filename for a given dataframe request."""
    cache_dir = Path(PATH_RESULTS) / "Coherence" / "Audio" / "Stats" / "Tidy dataframes"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = {"subject": subject, "freq": freq, "joints": joints, "kinematic_measure": kinematic_measure,
               "acoustic_measure": acoustic_measure, "lag": lag, "average": average, "value_type": value_type,
               "condition": condition, "use_condition_results": use_condition_results, }

    key = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]

    return cache_dir / f"coherence_tidy_{average}_{value_type}_{key}.parquet"


def _partial_eta_squared(anova_table, term):
    """Partial eta squared = SS_effect / (SS_effect + SS_error)."""
    if term is None or "Residual" not in anova_table.index:
        return np.nan

    ss_effect = anova_table.loc[term, "sum_sq"]
    ss_error = anova_table.loc["Residual", "sum_sq"]

    return ss_effect / (ss_effect + ss_error)


def print_colored_anova_summary(res, compare, value_col="dv", gradient=("#d9d9d9", "#fee08b", "#f46d43", "#a50026"),
                                decimals=6, global_scale=True, sort_means=True, truncate=None):
    """
    Print ANOVA effects and colored group means.

    Parameters
    ----------
    res : dict
        Output dictionary returned by coherence_anova().
        Must contain:
        - "anova_table"
        - "anova_dataframe"

    compare : str or tuple/list[str]
        Same compare argument used in coherence_anova().
        Can use aliases such as "freq", "joints", "acoustic_measures".

    value_col : str
        Column to summarize, usually "dv".

    gradient : list[str or tuple]
        Color gradient for coherence values. Low values map to first color,
        high values to last color.

    decimals : int
        Number of decimals for means.

    global_scale : bool
        If True, all means are colored using one common min/max scale.
        If False, each section gets its own scale.

    sort_means : bool
        If True, group means are printed from highest to lowest.
    """
    anova_table = res["anova_table"]
    anova_df = res["anova_dataframe"]

    if isinstance(compare, str):
        compare = (compare,)
    else:
        compare = tuple(compare)

    # Convert aliases to actual dataframe columns.
    compare_cols = tuple(COMPARE_TO_COLUMN.get(c, c) for c in compare)

    # Global color scale across all printed means.
    if global_scale:
        all_values = []

        for i in range(1, len(compare_cols) + 1):
            for comb in combinations(compare_cols, i):
                means = anova_df.groupby(list(comb), observed=True)[value_col].mean()
                all_values.extend(means.dropna().to_numpy())

        vmin_global = float(np.min(all_values))
        vmax_global = float(np.max(all_values))

    print("ANOVA formula")
    print("-" * 13)
    print(res.get("formula", "Formula unavailable"))
    print()

    for i in range(1, len(compare_cols) + 1):
        for comb in combinations(compare_cols, i):
            comb = list(comb)
            title = " × ".join(comb)

            print(title)
            print("=" * len(title))

            term = _anova_term_for_columns(comb, anova_table)

            if term is not None:
                row = anova_table.loc[term]

                df_effect = row.get("df", np.nan)
                sum_sq = row.get("sum_sq", np.nan)
                f_value = row.get("F", np.nan)
                p_value = row.get("PR(>F)", np.nan)
                eta_p2 = _partial_eta_squared(anova_table, term)

                print(f"Effect: {term}\n"
                      f"  df = {df_effect:.0f}\n"
                      f"  SS = {sum_sq:.6f}\n"
                      f"  F = {f_value:.4f}\n"
                      f"  p = {_p_colored(p_value)} {_p_to_stars(p_value)}\n"
                      f"  partial η² = {eta_p2:.4f}")
            else:
                print("No matching ANOVA term found for this combination.")

            print()

            summary = (anova_df.groupby(comb, observed=True)[value_col].agg(["mean", "std", "count"]).reset_index())

            summary["sem"] = summary["std"] / np.sqrt(summary["count"])

            if sort_means:
                summary = summary.sort_values("mean", ascending=False)

            if global_scale:
                vmin, vmax = vmin_global, vmax_global
            else:
                vmin = float(summary["mean"].min())
                vmax = float(summary["mean"].max())

            print(f"{'Level':<66} {'Mean':>12} {'SD':>12} {'SEM':>12} {'n':>6}")
            print("-" * 118)

            row_idx = 0

            for _, row in summary.iterrows():
                label = " × ".join(str(row[col]) for col in comb)

                mean_value = row["mean"]
                rgb = _interpolate_gradient(mean_value, vmin, vmax, gradient)
                mean_str = _rgb_text(f"{mean_value:.{decimals}f}", rgb)

                sd_str = f"{row['std']:.{decimals}f}"
                sem_str = f"{row['sem']:.{decimals}f}"
                n_str = f"{int(row['count'])}"

                print(f"{label:<70} {mean_str:>12} {sd_str:>12} {sem_str:>12} {n_str:>6}")

                row_idx += 1
                if truncate is not None and row_idx > truncate:
                    print(f"... and {len(summary) - truncate} more.")
                    break

            print()


# ---------------------------------------------------------------------
# 1. Reusable tidy dataframe builder
# ---------------------------------------------------------------------

def build_coherence_tidy_dataframe(subject, freq, joints, kinematic_measure, acoustic_measure, lag, average="subject",
                                   dataframe=None, cache_path=None, force_rebuild=False, save_cache=True,
                                   value_type="averages", series_value="Full dataset", condition=None,
                                   use_condition_results=False, freq_atol=None, missing="raise", ):
    """
    Build or load a tidy dataframe from Krajjat coherence result files.

    Parameters
    ----------
    subject : str or list[str]
        Subject ID(s).

    freq : float, tuple, list, or None
        Frequency selection.
        - float: nearest available frequency is selected.
        - tuple: lower/upper frequency band, averaged within the band.
        - list: mixture of floats and tuples.
        - None: all frequencies are kept separately.

    joints : str or list[str]
        Joint(s), e.g. "HandRight" or ["HandRight", "HandLeft"].

    kinematic_measure : str or list[str]
        Kinematic measure(s), e.g. "velocity", "acceleration".

    acoustic_measure : str or list[str]
        Acoustic measure(s), e.g. "envelope", "pitch", "intensity".

    lag : float, list[float], or None
        Lag(s) in seconds. If None, all lags found in each file are used.

    average : {"subject", "trial"}
        Determines file suffix:
        - "subject": no suffix
        - "trial": "_avgtrial"

    dataframe : pd.DataFrame or None
        If provided, returned directly. This lets later analysis functions avoid rebuilding.

    cache_path : str or Path or None
        If provided and the file exists, the dataframe is loaded unless force_rebuild=True.
        If None, a deterministic cache path is created from the parameters.

    force_rebuild : bool
        If True, ignore cached dataframe and rebuild from JSON files.

    save_cache : bool
        If True, save the built dataframe to cache_path.

    value_type : {"averages", "z_scores"}
        Which result container to use.

    series_value : str
        Usually "Full dataset".

    freq_atol : float or None
        Optional tolerance for point-frequency matching.

    missing : {"raise", "skip"}
        Whether to raise an error or skip missing files/entries.

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with one row per selected:
        subject × kinematic_measure × acoustic_measure × joint × lag × frequency.
    """
    if dataframe is not None:
        return dataframe.copy()

    if average not in {"subject", "trial"}:
        raise ValueError("`average` must be either 'subject' or 'trial'.")

    if value_type not in {"averages", "z_scores"}:
        raise ValueError("`value_type` must be either 'averages' or 'z_scores'.")

    if missing not in {"raise", "skip"}:
        raise ValueError("`missing` must be either 'raise' or 'skip'.")

    if cache_path is None:
        cache_path = _make_cache_path(subject=subject, freq=freq, joints=joints, kinematic_measure=kinematic_measure,
                                      acoustic_measure=acoustic_measure, lag=lag, average=average,
                                      value_type=value_type, condition=condition,
                                      use_condition_results=use_condition_results, )
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    subjects = _as_list(subject, "subject")
    joints = _as_list(joints, "joints")
    kinematic_measures = _as_list(kinematic_measure, "kinematic_measure")
    acoustic_measures = _as_list(acoustic_measure, "acoustic_measure")
    requested_lags = _as_list(lag, "lag")
    requested_conditions = _as_list(condition, "condition")
    freq_specs = _normalize_freq_specs(freq)

    average_string = "_concat" if average == "subject" else "_avgtrial"

    rows = []
    skipped = []

    for subj in subjects:
        for kin in kinematic_measures:
            for ac in acoustic_measures:
                coherence_subfolder = "Conditions" if use_condition_results else "Audio"
                folder = op.join(PATH_RESULTS, "Coherence", coherence_subfolder, "Per subject", f"{kin} x {ac}")

                filename = f"coh_{SYSTEM.lower()}_fs{int(FS)}_{kin}_{ac}_{subj}{average_string}{REREF_SUFFIX}.gzip"
                path_json = op.join(folder, filename)

                if not op.exists(path_json):
                    msg = f"Missing file: {path_json}"
                    if missing == "raise":
                        raise FileNotFoundError(msg)
                    skipped.append(msg)
                    continue

                out = Results.load(path_json)

                container = getattr(out, value_type)

                try:
                    target_key = _find_target_measure_key(container, ac)
                except KeyError as e:
                    if missing == "raise":
                        raise
                    skipped.append(str(e))
                    continue

                frequencies = np.asarray(out.frequencies, dtype=float)

                # If lag is None, use all lags available in this file.
                if requested_lags is None:
                    lags_to_use = list(out.lags)
                else:
                    lags_to_use = requested_lags

                if use_condition_results:
                    if requested_conditions is None:
                        series_values_to_use = list(out.series_values)
                    else:
                        series_values_to_use = requested_conditions
                else:
                    series_values_to_use = [series_value]

                for current_series_value in series_values_to_use:
                    if current_series_value not in container[target_key][kin]:
                        msg = (f"Series value {current_series_value!r} not found in {path_json}. "
                               f"Available series values: {list(container[target_key][kin].keys())}")
                        if missing == "raise":
                            raise KeyError(msg)
                        skipped.append(msg)
                        continue

                    current_condition = current_series_value if use_condition_results else np.nan

                    for joint in joints:
                        for requested_lag in lags_to_use:
                            try:
                                lag_dict = container[target_key][kin][current_series_value][joint]
                                lag_key = _find_matching_key(lag_dict, requested_lag, name="lag", atol=1e-8)
                                spectrum = lag_dict[lag_key]

                            except Exception as e:
                                if missing == "raise":
                                    raise
                                skipped.append(f"Skipped {subj}, {kin}, {ac}, {current_series_value}, {joint}, "
                                               f"lag={requested_lag}: {e}")
                                continue

                            selected_rows = _select_frequency_values(values=spectrum, frequencies=frequencies,
                                                                     freq_specs=freq_specs, freq_atol=freq_atol)

                            for r in selected_rows:
                                rows.append({"subject": subj, "condition": current_condition, "kinematic_measure": kin,
                                             "acoustic_measure": ac, "joint": joint, "lag": float(lag_key),
                                             "requested_lag": np.nan if requested_lag is None else float(requested_lag),
                                             "freq_label": r["freq_label"], "freq_kind": r["freq_kind"],
                                             "frequency": r["frequency"], "freq_lower": r["freq_lower"],
                                             "freq_upper": r["freq_upper"], "value_type": value_type,
                                             "coherence": r["value"], "source_file": path_json, })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No rows were created. Check subject names, joints, measures, lags, "
                         "frequency selection, and file paths.")

    df.attrs["skipped"] = skipped

    if save_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        df.attrs["cache_path"] = str(cache_path)

    return df


# ---------------------------------------------------------------------
# 2. ANOVA function
# ---------------------------------------------------------------------

def coherence_anova(subject, freq, joints, kinematic_measure, acoustic_measure, lag, compare, average="subject",
                    dataframe=None, cache_path=None, force_rebuild=False, save_cache=True, value_type="averages",
                    series_value="Full dataset", condition=None, freq_atol=None, missing="raise", transform=None, typ=2,
                    include_subject_block=True, aggregate_func="mean", ):
    """
    Run an ANOVA on coherence values.

    This function first obtains a tidy dataframe using
    build_coherence_tidy_dataframe(), unless a dataframe is already supplied.

    Parameters
    ----------
    compare : str or tuple/list[str]
        Level(s) to compare. Allowed:
        - "joints"
        - "freq"
        - "kinematic_measures"
        - "acoustic_measures"
        - "lags"
        - "subject"

        Examples:
        compare="joints"
        compare=("joints", "acoustic_measures")
        compare=("freq", "acoustic_measures", "joints")

    transform : {None, "logit"}
        Optional transformation of coherence.
        - None: use raw coherence.
        - "logit": apply logit transform after clipping to avoid 0/1.

    typ : {1, 2, 3}
        Type of sums of squares passed to statsmodels.stats.anova.anova_lm.

    include_subject_block : bool
        If True and subject is not itself a comparison factor, include C(subject)
        as an additive blocking factor.

    aggregate_func : str
        Aggregation function used to ensure one value per cell before ANOVA.
        Usually "mean".

    Returns
    -------
    dict
        {
            "anova_table": ANOVA table,
            "model": fitted OLS model,
            "formula": model formula,
            "dataframe": original tidy dataframe,
            "anova_dataframe": aggregated dataframe used for ANOVA
        }
    """
    compare = _as_compare_tuple(compare)

    factor_cols = [COMPARE_TO_COLUMN[c] for c in compare]
    use_condition_results = ("condition" in compare) or (condition is not None)

    df = build_coherence_tidy_dataframe(subject=subject, freq=freq, joints=joints, kinematic_measure=kinematic_measure,
                                        acoustic_measure=acoustic_measure, lag=lag, average=average,
                                        dataframe=dataframe, cache_path=cache_path, force_rebuild=force_rebuild,
                                        save_cache=save_cache, value_type=value_type, series_value=series_value,
                                        condition=condition, use_condition_results=use_condition_results,
                                        freq_atol=freq_atol, missing=missing, )

    df = df.copy()

    if transform is None:
        df["dv"] = df["coherence"]

    elif transform == "logit":
        # Coherence is bounded [0, 1]. Clip to avoid infinite values.
        eps = 1e-6
        df["dv"] = logit(np.clip(df["coherence"], eps, 1 - eps))

    else:
        raise ValueError("`transform` must be None or 'logit'.")

    df = df.dropna(subset=["dv"])

    # Make sure the requested factors actually vary.
    for col in factor_cols:
        n_levels = df[col].nunique(dropna=True)
        if n_levels < 2:
            raise ValueError(f"Cannot compare {col!r}: only {n_levels} level found. "
                             f"Available values: {sorted(df[col].dropna().unique())}")

    # Aggregate away dimensions that are not part of the requested comparison.
    # Always keep subject unless subject is absent and there is only one subject anyway.
    group_cols = list(dict.fromkeys(["subject"] + factor_cols))

    anova_df = (df.groupby(group_cols, observed=True)["dv"].agg(aggregate_func).reset_index())

    # Build formula.
    comparison_term = " * ".join([f"C({col})" for col in factor_cols])

    subject_is_factor = "subject" in factor_cols
    has_multiple_subjects = anova_df["subject"].nunique() > 1

    if (include_subject_block and has_multiple_subjects and not subject_is_factor):
        formula = f"dv ~ C(subject) + {comparison_term}"
    else:
        formula = f"dv ~ {comparison_term}"

    model = ols(formula, data=anova_df).fit()
    anova_table = anova_lm(model, typ=typ)

    return {"anova_table": anova_table, "model": model, "formula": formula, "dataframe": df,
            "anova_dataframe": anova_df, }


def _format_reference(value):
    """Format a reference level value for Patsy's Treatment() contrast."""
    if isinstance(value, str):
        return f'"{value.replace(chr(34), chr(92) + chr(34))}"'
    return repr(value)


def _term_to_formula(term, continuous_cols=frozenset(), reference_levels=None):
    """Convert ('freq_label', 'joint') to 'C(freq_label):C(joint)'."""
    if reference_levels is None:
        reference_levels = {}
    parts = []
    for col in term:
        if col in continuous_cols:
            parts.append(col)
        elif col in reference_levels:
            parts.append(f"C({col}, Treatment({_format_reference(reference_levels[col])}))")
        else:
            parts.append(f"C({col})")
    return ":".join(parts)


def _expand_term(term, continuous_cols, reference_levels=None):
    """
    Expand a term tuple into one or more Patsy formula strings.

    For all-categorical terms, returns one string identical to _term_to_formula().
    For terms with one continuous predictor of degree d, returns d strings — one per
    polynomial power — each crossed with any categorical parts of the term.
    Terms with multiple continuous predictors use degree-1 only (no polynomial cross-terms).
    """
    cat_cols = [col for col in term if col not in continuous_cols]
    cont_cols = [(col, continuous_cols[col]) for col in term if col in continuous_cols]
    cat_str = ":".join(
        f"C({col}, Treatment({_format_reference(reference_levels[col])}))" if reference_levels and col in reference_levels else f"C({col})"
        for col in cat_cols)

    if not cont_cols:
        return [cat_str] if cat_str else []

    result = []
    if len(cont_cols) == 1:
        col, degree = cont_cols[0]
        for d in range(1, degree + 1):
            cont_str = col if d == 1 else f"I({col}**{d})"
            result.append(f"{cont_str}:{cat_str}" if cat_str else cont_str)
    else:
        cont_str = ":".join(col for col, _ in cont_cols)
        result.append(f"{cont_str}:{cat_str}" if cat_str else cont_str)

    return result


def _make_lmm_formula(factor_cols, drop_term=None, drop_predictor=None, continuous_cols=None, reference_levels=None):
    """
    Build an explicit fixed-effects formula.

    drop_term : tuple or None
        Drop a specific term tuple (used by categorical LRTs, current behaviour).
    drop_predictor : str or None
        Drop all terms that contain this column (used by continuous LRTs).
    continuous_cols : dict {col_name: degree} or None
        Columns to treat as continuous with polynomial expansion.
        If None or empty, all terms are wrapped in C() as before.
    reference_levels : dict {col_name: value} or None
        Reference level per categorical column, expressed as the actual data value.
        If None, Patsy uses alphabetical order (default behaviour).
    """
    if continuous_cols is None:
        continuous_cols = {}
    if reference_levels is None:
        reference_levels = {}

    terms = _make_fixed_effect_terms(factor_cols)

    if drop_term is not None:
        terms = [t for t in terms if tuple(t) != tuple(drop_term)]

    if drop_predictor is not None:
        terms = [t for t in terms if drop_predictor not in t]

    if not terms:
        return "dv ~ 1"

    if continuous_cols:
        formula_parts = []
        for term in terms:
            formula_parts.extend(_expand_term(term, continuous_cols, reference_levels=reference_levels))
    else:
        formula_parts = [_term_to_formula(term, continuous_cols=continuous_cols, reference_levels=reference_levels) for
                         term in terms]

    if not formula_parts:
        return "dv ~ 1"

    return f"dv ~ {' + '.join(formula_parts)}"


def _fit_mixedlm_with_fallbacks(formula, data, random_group="subject", re_formula="1", vc_formula=None, reml=False,
                                methods=("lbfgs", "powell", "cg"), maxiter=1000, ):
    """
    Fit a MixedLM model, trying several optimizers if needed.
    """
    last_error = None
    last_result = None

    for method in methods:
        try:
            model = mixedlm(formula, data=data, groups=data[random_group], re_formula=re_formula,
                            vc_formula=vc_formula, )

            result = model.fit(reml=reml, method=method, maxiter=maxiter, disp=False, )

            last_result = result

            if getattr(result, "converged", False):
                return result

        except Exception as e:
            last_error = e

    if last_result is not None:
        print("Warning: MixedLM did not fully converge with the attempted optimizers.")
        return last_result

    raise RuntimeError(f"MixedLM failed to fit. Last error was: {last_error}")


def _lmm_likelihood_ratio_tests(full_result, factor_cols, data, random_group="subject", re_formula="1", vc_formula=None,
                                methods=("lbfgs", "powell", "cg"), maxiter=1000, continuous_cols=None,
                                reference_levels=None):
    """
    Run likelihood-ratio tests by comparing the full model to reduced models.

    For all-categorical models: one test per term tuple (main effects and interactions
    separately), identical to the previous behaviour.
    For models with continuous predictors: one test per predictor column, dropping all
    terms involving that predictor (main effect + polynomial terms + interactions).

    Use reml=False for these tests.
    """
    if continuous_cols is None:
        continuous_cols = {}

    rows = []
    full_llf = full_result.llf
    full_n_fe = len(full_result.fe_params)

    if continuous_cols:
        test_items = [(col, {"drop_predictor": col}) for col in factor_cols]
    else:
        test_items = [(_term_to_formula(term, continuous_cols=continuous_cols, reference_levels=reference_levels),
                       {"drop_term": term}) for term in _make_fixed_effect_terms(factor_cols)]

    for label, drop_kwargs in test_items:
        reduced_formula = _make_lmm_formula(factor_cols, continuous_cols=continuous_cols,
                                            reference_levels=reference_levels, **drop_kwargs)

        try:
            reduced_result = _fit_mixedlm_with_fallbacks(formula=reduced_formula, data=data, random_group=random_group,
                                                         re_formula=re_formula, vc_formula=vc_formula, reml=False,
                                                         methods=methods, maxiter=maxiter, )

            reduced_llf = reduced_result.llf
            reduced_n_fe = len(reduced_result.fe_params)
            df_diff = full_n_fe - reduced_n_fe
            lr_stat = 2 * (full_llf - reduced_llf)

            if df_diff <= 0 or not np.isfinite(full_llf) or not np.isfinite(reduced_llf) or lr_stat < 0:
                lr_stat = np.nan
                p_value = np.nan
            else:
                p_value = chi2.sf(lr_stat, df_diff)

            rows.append({"term": label, "df": df_diff, "LR": lr_stat, "p": p_value, "full_llf": full_llf,
                         "reduced_llf": reduced_llf, "reduced_formula": reduced_formula,
                         "converged": getattr(reduced_result, "converged", np.nan), })

        except Exception as e:
            rows.append(
                {"term": label, "df": np.nan, "LR": np.nan, "p": np.nan, "full_llf": full_llf, "reduced_llf": np.nan,
                 "reduced_formula": reduced_formula, "converged": False, "error": str(e), })

    return pd.DataFrame(rows)


def coherence_lmm(subject, freq, joints, kinematic_measure, acoustic_measure, lag, compare, average="subject",
                  dataframe=None, cache_path=None, force_rebuild=False, save_cache=True, value_type="averages",
                  series_value="Full dataset", condition=None, freq_atol=None, missing="raise", transform=None,
                  aggregate_func="mean", random_group="subject", re_formula="1", vc_formula=None, reml=False,
                  run_lrt=True, methods=("lbfgs", "powell", "cg"), maxiter=1000, continuous=None, ):
    """
    Run a linear mixed model on coherence values.

    This is the LMM equivalent of coherence_anova(), using the same tidy
    dataframe builder.

    Parameters
    ----------
    compare : str or tuple/list[str]
        Fixed-effect factors to include. Allowed values are the same as
        for coherence_anova(), except that "subject" should usually not
        be included because subject is modeled as a random effect.

    random_group : str
        Grouping variable for the random intercept. Usually "subject".

    re_formula : str
        Random-effect formula. Default "1" means random intercept only.

    vc_formula : dict or None
        Optional variance-component formulas for additional random effects.
        Leave as None for the first version.

    reml : bool
        Use False if running likelihood-ratio tests.
        Use True only for final parameter estimation without model comparison.

    run_lrt : bool
        If True, run likelihood-ratio tests for each fixed-effect term.

    continuous : dict {compare_alias: degree} or None
            Factors to treat as continuous predictors rather than categorical.
            Keys are compare aliases (e.g. "lags"), values are polynomial degree.
            - degree=1 : linear term only.
            - degree=2 : linear + quadratic (recommended for lag).
            None means all factors are categorical (default behaviour).
            Example: continuous={"lags": 2}

    Returns
    -------
    dict
        {
            "model": fitted MixedLM result,
            "formula": fixed-effects formula,
            "lrt_table": likelihood-ratio table or None,
            "dataframe": original tidy dataframe,
            "lmm_dataframe": aggregated dataframe used for LMM
        }
    """
    compare = _as_compare_tuple(compare)

    if "subject" in compare and random_group == "subject":
        raise ValueError("`subject` should usually not be included in compare for an LMM, "
                         "because it is already modeled as a random effect.")

    factor_cols = [COMPARE_TO_COLUMN[c] for c in compare]

    continuous_cols = {}
    if continuous is not None:
        for alias, degree in continuous.items():
            col = COMPARE_TO_COLUMN.get(alias, alias)
            if col not in factor_cols:
                raise ValueError(f"Continuous predictor {alias!r} (column {col!r}) is not in compare={compare}. "
                                 f"A predictor must appear in compare to be treated as continuous.")
            continuous_cols[col] = int(degree)

    # Build reference levels from the first element of each input list.
    # This ensures C(col) uses the user's list order rather than alphabetical.
    _first = {"kinematic_measure": (_as_list(kinematic_measure) or [None])[0],
              "acoustic_measure": (_as_list(acoustic_measure) or [None])[0], "joint": (_as_list(joints) or [None])[0],
              "lag": (_as_list(lag) or [None])[0], "subject": (_as_list(subject) or [None])[0],
              "condition": (_as_list(condition) or [None])[0],
              "freq_label": _normalize_freq_specs(freq)[0]["label"] if freq is not None else None, }
    reference_levels = {col: _first[col] for col in factor_cols if
                        col in _first and _first[col] is not None and col not in continuous_cols}

    use_condition_results = ("condition" in compare) or (condition is not None)

    df = build_coherence_tidy_dataframe(subject=subject, freq=freq, joints=joints, kinematic_measure=kinematic_measure,
                                        acoustic_measure=acoustic_measure, lag=lag, average=average,
                                        dataframe=dataframe, cache_path=cache_path, force_rebuild=force_rebuild,
                                        save_cache=save_cache, value_type=value_type, series_value=series_value,
                                        condition=condition, use_condition_results=use_condition_results,
                                        freq_atol=freq_atol, missing=missing, )

    df = df.copy()

    if transform is None:
        df["dv"] = df["coherence"]

    elif transform == "logit":
        eps = 1e-6
        df["dv"] = logit(np.clip(df["coherence"], eps, 1 - eps))

    else:
        raise ValueError("`transform` must be None or 'logit'.")

    df = df.dropna(subset=["dv"])

    # Make sure the requested fixed-effect factors vary.
    for col in factor_cols:
        n_levels = df[col].nunique(dropna=True)
        if n_levels < 2:
            raise ValueError(f"Cannot compare {col!r}: only {n_levels} level found. "
                             f"Available values: {sorted(df[col].dropna().unique())}")

    if random_group not in df.columns:
        raise ValueError(f"`random_group={random_group}` is not a column in the dataframe.")

    # Aggregate away dimensions that are not part of the requested model.
    group_cols = list(dict.fromkeys([random_group] + factor_cols))

    lmm_df = (df.groupby(group_cols, observed=True)["dv"].agg(aggregate_func).reset_index())

    # MixedLM cannot handle missing grouping values.
    lmm_df = lmm_df.dropna(subset=[random_group, "dv"])

    formula = _make_lmm_formula(factor_cols, continuous_cols=continuous_cols, reference_levels=reference_levels)

    model_result = _fit_mixedlm_with_fallbacks(formula=formula, data=lmm_df, random_group=random_group,
                                               re_formula=re_formula, vc_formula=vc_formula, reml=reml, methods=methods,
                                               maxiter=maxiter, )

    if run_lrt:
        if reml:
            print("Warning: LRTs should be run with reml=False. Refitting full model with reml=False.")
            model_result_for_lrt = _fit_mixedlm_with_fallbacks(formula=formula, data=lmm_df, random_group=random_group,
                                                               re_formula=re_formula, vc_formula=vc_formula, reml=False,
                                                               methods=methods, maxiter=maxiter, )
        else:
            model_result_for_lrt = model_result

        lrt_table = _lmm_likelihood_ratio_tests(full_result=model_result_for_lrt, factor_cols=factor_cols, data=lmm_df,
                                                random_group=random_group, re_formula=re_formula, vc_formula=vc_formula,
                                                methods=methods, maxiter=maxiter, continuous_cols=continuous_cols,
                                                reference_levels=reference_levels, )
    else:
        lrt_table = None

    return {"model": model_result, "formula": formula, "lrt_table": lrt_table, "dataframe": df,
            "lmm_dataframe": lmm_df, }


def print_anova_results(res, subject, freq, joints, kinematic_measure, acoustic_measure, lag, compare, truncate):
    compare_corrected = get_compare_corrected(compare)

    print("           _   _  ______      __     ")
    print("     /\\   | \\ | |/ __ \\ \\    / /\\     ")
    print("    /  \\  |  \\| | |  | \\ \\  / /  \\    ")
    print("   / /\\ \\ | . ` | |  | |\\ \\/ / /\\ \\   ")
    print("  / ____ \\| |\\  | |__| | \\  / ____ \\  ")
    print(" /_/    \\_\\_| \\_|\\____/   \\/_/    \\_\\ ")
    print("")
    print(res["formula"])

    print("\n==================== FACTORS ====================")
    subject_text = ", ".join(subject)
    freq_text = ", ".join(str(f) for f in freq)
    joints_text = ", ".join(joints)
    print(f"Subjects: {len(subject)}: {subject_text}")
    print(f"Frequencies: {len(freq)}: {freq_text}")
    print(f"Joints: {len(joints)}: {joints_text}")
    print(f"Kinematic measures: {len(kinematic_measure)}: {', '.join(kinematic_measure)}")
    print(f"Acoustic measures: {len(acoustic_measure)}: {', '.join(acoustic_measure)}")
    lag_text = ', '.join(map(str, lag))
    print(f"Lags: {len(lag)}: {lag_text}")
    if "condition" in res["dataframe"].columns and res["dataframe"]["condition"].notna().any():
        conditions_found = sorted(res["dataframe"]["condition"].dropna().unique())
        print(f"Conditions: {len(conditions_found)}: {', '.join(map(str, conditions_found))}")

    print("\n==================== RESULTS ====================")
    print(res["anova_table"])

    print("\n==================== SUMMARY ====================")
    print_colored_anova_summary(res, compare=compare_corrected, value_col="dv",
                                gradient=["#40E0D0", "#99CC00", "#FFCC00", "#FFA500", "#FF0000"], decimals=6,
                                truncate=truncate)


def _format_stat(value, decimals=6):
    """Format finite and non-finite numeric values consistently."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def _bool_colored(value):
    """Color boolean-like diagnostic values."""
    if bool(value):
        return _rgb_text(str(value), (0, 130, 0))
    return _rgb_text(str(value), (180, 0, 0))


def _finite_colored(value, decimals=6):
    """Color finite values green and non-finite values red."""
    text = _format_stat(value, decimals=decimals)

    try:
        finite = np.isfinite(float(value))
    except (TypeError, ValueError):
        finite = False

    if finite:
        return _rgb_text(text, (0, 130, 0))
    return _rgb_text(text, (180, 0, 0))


def _signed_value_colored(value, max_abs=None, decimals=6):
    """Color signed coefficients using a blue/grey/red diverging scale."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(value):
        return _rgb_text("nan", (150, 150, 150))

    if max_abs is None or max_abs == 0 or np.isnan(max_abs):
        max_abs = abs(value) if value != 0 else 1.0

    rgb = _interpolate_gradient(value, -max_abs, max_abs, ["#40E0D0", "#D9D9D9", "#FF0000"], )

    return _rgb_text(f"{value:.{decimals}f}", rgb)


def _positive_value_colored(value, vmin, vmax, gradient, decimals=6):
    """Color positive/magnitude values using the same coherence gradient as ANOVA."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(value):
        return _rgb_text("nan", (150, 150, 150))

    rgb = _interpolate_gradient(value, vmin, vmax, gradient)
    return _rgb_text(f"{value:.{decimals}f}", rgb)


def _get_fixed_effects_table(result):
    """Return a tidy table of fixed-effect estimates from a MixedLM result."""
    terms = list(result.fe_params.index)

    table = pd.DataFrame({"term": terms, "coef": result.fe_params.to_numpy(), })

    if hasattr(result, "bse"):
        table["std_err"] = result.bse.reindex(terms).to_numpy()
    elif hasattr(result, "bse_fe"):
        table["std_err"] = np.asarray(result.bse_fe)
    else:
        table["std_err"] = np.nan

    if hasattr(result, "tvalues"):
        table["z"] = result.tvalues.reindex(terms).to_numpy()
    else:
        table["z"] = table["coef"] / table["std_err"]

    if hasattr(result, "pvalues"):
        table["p"] = result.pvalues.reindex(terms).to_numpy()
    else:
        table["p"] = np.nan

    return table


def print_colored_lmm_fixed_effects(result, decimals=6, truncate=40, sort_by=None):
    """
    Print fixed-effect estimates from a fitted MixedLM result.

    Coefficients are colored by sign/magnitude. P-values are colored by
    conventional thresholds.
    """
    table = _get_fixed_effects_table(result)

    if sort_by == "p":
        table = table.sort_values("p", na_position="last")
    elif sort_by == "abs_coef":
        table = table.assign(abs_coef=table["coef"].abs()).sort_values("abs_coef", ascending=False)

    finite_coefs = table["coef"].replace([np.inf, -np.inf], np.nan).dropna()
    max_abs = float(finite_coefs.abs().max()) if len(finite_coefs) else 1.0

    print("\n==================== FIXED EFFECTS ====================")
    print(f"{'Term':<85} {'Coef.':>14} {'Std.Err.':>12} {'z':>10} {'p':>14}")
    print("-" * 140)

    for idx, (_, row) in enumerate(table.iterrows()):
        if truncate is not None and idx > truncate:
            print(f"... and {len(table) - truncate} more.")
            break

        term = str(row["term"])
        coef = _signed_value_colored(row["coef"], max_abs=max_abs, decimals=decimals)
        std_err = _format_stat(row["std_err"], decimals=decimals)
        z_value = _format_stat(row["z"], decimals=4)
        p_value = _p_colored(row["p"])
        stars = _p_to_stars(row["p"]) if not np.isnan(row["p"]) else ""

        print(f"{term:<85} {coef:>14} {std_err:>12} {z_value:>10} {p_value:>14} {stars}")


def print_colored_lmm_lrt_table(lrt_table, decimals=6):
    """Print colored likelihood-ratio tests."""
    if lrt_table is None or len(lrt_table) == 0:
        print("No likelihood-ratio tests were run.")
        return

    print(f"{'Term':<70} {'df':>6} {'LR':>14} {'p':>14} {'converged':>12}")
    print("-" * 125)

    for _, row in lrt_table.iterrows():
        term = str(row.get("term", ""))
        df_value = row.get("df", np.nan)
        lr_value = row.get("LR", np.nan)
        p_value = row.get("p", np.nan)
        converged = row.get("converged", np.nan)

        df_str = _format_stat(df_value, decimals=0)
        lr_str = _format_stat(lr_value, decimals=decimals)
        p_str = _p_colored(p_value)
        if isinstance(converged, (bool, np.bool_)):
            converged_str = _bool_colored(converged)
        else:
            converged_str = str(converged)

        stars = _p_to_stars(p_value) if not np.isnan(p_value) else ""

        print(f"{term:<70} {df_str:>6} {lr_str:>14} {p_str:>14} {converged_str:>12} {stars}")


def print_colored_lmm_marginal_means(res, compare, value_col="dv",
                                     gradient=("#40E0D0", "#99CC00", "#FFCC00", "#FFA500", "#FF0000"), decimals=6,
                                     global_scale=True, sort_means=True, truncate=20, ):
    """Print colored marginal means for LMM input data."""
    lmm_df = res["lmm_dataframe"]

    if isinstance(compare, str):
        compare = (compare,)
    else:
        compare = tuple(compare)

    compare_cols = tuple(COMPARE_TO_COLUMN.get(c, c) for c in compare)

    if global_scale:
        all_values = []
        for i in range(1, len(compare_cols) + 1):
            for comb in combinations(compare_cols, i):
                means = lmm_df.groupby(list(comb), observed=True)[value_col].mean()
                all_values.extend(means.dropna().to_numpy())

        if len(all_values) == 0:
            vmin_global, vmax_global = 0.0, 1.0
        else:
            vmin_global = float(np.min(all_values))
            vmax_global = float(np.max(all_values))

    for i in range(1, len(compare_cols) + 1):
        for comb in combinations(compare_cols, i):
            comb = list(comb)
            title = " × ".join(comb)

            print(title)
            print("=" * len(title))

            summary = (lmm_df.groupby(comb, observed=True)[value_col].agg(["mean", "std", "count"]).reset_index())

            summary["sem"] = summary["std"] / np.sqrt(summary["count"])

            if sort_means:
                summary = summary.sort_values("mean", ascending=False)

            if global_scale:
                vmin, vmax = vmin_global, vmax_global
            else:
                vmin = float(summary["mean"].min())
                vmax = float(summary["mean"].max())

            print(f"{'Level':<66} {'Mean':>12} {'SD':>12} {'SEM':>12} {'n':>6}")
            print("-" * 118)

            for idx, (_, row) in enumerate(summary.iterrows()):
                if truncate is not None and idx > truncate:
                    print(f"... and {len(summary) - truncate} more.")
                    break

                label = " × ".join(str(row[col]) for col in comb)
                mean_str = _positive_value_colored(row["mean"], vmin=vmin, vmax=vmax, gradient=gradient,
                                                   decimals=decimals, )

                sd_str = _format_stat(row["std"], decimals=decimals)
                sem_str = _format_stat(row["sem"], decimals=decimals)
                n_str = f"{int(row['count'])}"

                print(f"{label:<70} {mean_str:>12} {sd_str:>12} {sem_str:>12} {n_str:>6}")

            print()


def print_lmm_results(res, compare, truncate=20, fixed_effects_truncate=40, show_full_summary=False,
                      show_fixed_effects=True, sort_fixed_effects_by=None, ):
    """Print LMM formula, diagnostics, optional model summary, fixed effects, LRTs, and colored group means."""
    compare_corrected = get_compare_corrected(compare)

    print(" _      __  __ __  __ ")
    print("| |    |  \\/  |  \\/  |")
    print("| |    | \\  / | \\  / |")
    print("| |___ | |\\/| | |\\/| |")
    print("|_____||_|  |_|_|  |_|")
    print("")
    print(res["formula"])

    if show_full_summary:
        print("\n==================== MODEL SUMMARY ====================")
        print(res["model"].summary())

    if show_fixed_effects:
        print_colored_lmm_fixed_effects(res["model"], decimals=6, truncate=fixed_effects_truncate,
                                        sort_by=sort_fixed_effects_by, )

    check_mixedlm_fit(res["model"])

    if res["lrt_table"] is not None:
        print("\n==================== LIKELIHOOD-RATIO TESTS ====================")
        print_colored_lmm_lrt_table(res["lrt_table"])

    print("\n==================== MARGINAL MEANS ====================")
    print_colored_lmm_marginal_means(res, compare=compare_corrected, value_col="dv",
                                     gradient=["#40E0D0", "#99CC00", "#FFCC00", "#FFA500", "#FF0000"], decimals=6,
                                     truncate=truncate, )


def check_mixedlm_fit(result, tol=1e-8):
    """Print basic colored diagnostics for a fitted MixedLM model."""
    print("\n==================== LMM DIAGNOSTICS ====================")

    converged = getattr(result, "converged", None)
    if converged is None:
        converged_text = "Unknown"
    else:
        converged_text = _bool_colored(converged)

    print(f"Converged: {converged_text}")
    print(f"Log-likelihood: {_finite_colored(result.llf)}")

    if hasattr(result, "scale"):
        print(f"Residual scale: {_finite_colored(result.scale)}")

    if hasattr(result, "cov_re") and result.cov_re is not None and result.cov_re.size > 0:
        print("\nRandom-effect covariance:")
        print(result.cov_re)

        min_var = np.min(np.diag(result.cov_re))
        if min_var < tol:
            min_var_text = _rgb_text(f"{min_var:.12g}", (180, 0, 0))
        else:
            min_var_text = _rgb_text(f"{min_var:.12g}", (0, 130, 0))

        print(f"\nMinimum random-effect variance: {min_var_text}")

        if min_var < tol:
            print(_rgb_text("\nWarning: the random-effect variance is approximately zero. "
                            "The model is effectively behaving like an ordinary fixed-effects model.", (180, 80, 0), ))

    if not np.isfinite(result.llf):
        print(_rgb_text("\nWarning: the log-likelihood is not finite. "
                        "Do not use likelihood-ratio tests for this model.", (180, 0, 0), ))


def get_compare_corrected(compare):
    compare_corrected = []

    for factor in compare:
        if factor in COMPARE_TO_COLUMN:
            compare_corrected.append(COMPARE_TO_COLUMN[factor])
        else:
            compare_corrected.append(factor)
    compare_corrected = tuple(compare_corrected)
    return compare_corrected


def run_anova(compare, subject=None, kinematic_measure=None, acoustic_measure=None, freq=None, joints=None, lag=None,
              average="subject", value_types="averages", transform=None, truncate=20):
    if subject is None:
        subject = SUBJECTS_NO_OUTLIERS
    if kinematic_measure is None:
        kinematic_measure = ["velocity", "acceleration", "jerk"]
    if acoustic_measure is None:
        acoustic_measure = DERIVATIVES
    if freq is None:
        freq = [(0.2, 0.4)]
    if joints is None:
        joints = list(UPPER_BODY_JOINTS)
    if lag is None:
        lag = [0.0]

    res = coherence_anova(subject=subject, freq=freq, joints=joints, kinematic_measure=kinematic_measure,
                          acoustic_measure=acoustic_measure, lag=lag, compare=compare, average=average,
                          value_type="averages", transform=None)
    print_anova_results(res, subject, freq, joints, kinematic_measure, acoustic_measure, lag, compare,
                        truncate=truncate)


def run_lmm(compare, subject=None, kinematic_measure=None, acoustic_measure=None, freq=None, joints=None, lag=None,
            average="subject", value_types="averages", transform=None, truncate=20, random_group="subject",
            re_formula="1", reml=False, run_lrt=False):
    if subject is None:
        subject = SUBJECTS_NO_OUTLIERS
    if kinematic_measure is None:
        kinematic_measure = ["velocity", "acceleration", "jerk"]
    if acoustic_measure is None:
        acoustic_measure = DERIVATIVES
    if freq is None:
        freq = [(0.2, 0.4)]
    if joints is None:
        joints = list(ALL_JOINTS)
    if lag is None:
        lag = [0.0]

    res = coherence_lmm(subject=subject, freq=freq, joints=joints, kinematic_measure=kinematic_measure,
                        acoustic_measure=acoustic_measure, lag=lag, compare=compare, average=average,
                        value_type=value_types, transform=transform, random_group=random_group, re_formula=re_formula,
                        reml=reml, run_lrt=run_lrt)

    print_lmm_results(res, compare, truncate=truncate)


def run_anova_lmm_digestible(compare, context_label, subject=None, kinematic_measure=None, acoustic_measure=None,
        freq=None, joints=None, lag=None, average="subject", value_type="averages", transform=None, truncate=10,
        random_group="subject", re_formula="1", reml=False, run_lrt=True, continuous=None,
        correct_pvalues=True, correction_alpha=0.05, correction_method="fdr_bh", format_output="console",
        sort_key=None, ):
    """
    Run ANOVA and LMM on the same parameter set, then print a compact
    side-by-side digestible summary.

    Important:
    - LMM p-values require run_lrt=True.
    - ANOVA and LMM must use the same joints/frequencies/measures/lags.

    correct_pvalues : bool, optional
        If True (default), also print the BH-corrected variant of the table
        (print_digestible_results_corrected), with the three added p-corr
        columns, correcting within THIS table only (see that function's
        docstring for exactly which p-values are pooled together). If False,
        only the original uncorrected table is printed.

    format_output : {"console", "tabs"}, optional
        Passed through to print_digestible_results_corrected() (only used
        when correct_pvalues=True). "console" (default) is the usual colored
        terminal table. "tabs" prints a plain tab-separated block instead,
        ready to select and paste directly into Excel -- see that function's
        docstring for the exact column layout and formatting differences.

    sort_key : callable or None, optional
        Passed through to both print_digestible_results() and
        print_digestible_results_corrected(). If given, called as
        sort_key(summary_dataframe) to reorder the level rows -- e.g.
        ascending by frequency instead of by coherence value -- without
        changing anything else about either table's layout ("Diff." row,
        Uncorrected/p-corrected columns, etc. stay exactly as they'd
        otherwise be). None (default) keeps the standard sort.
    """
    if subject is None:
        subject = SUBJECTS_NO_OUTLIERS
    if kinematic_measure is None:
        kinematic_measure = ["velocity", "acceleration", "jerk"]
    if acoustic_measure is None:
        acoustic_measure = DERIVATIVES
    if freq is None:
        freq = [(0.2, 0.4)]
    if joints is None:
        joints = list(UPPER_BODY_JOINTS)
    if lag is None:
        lag = [0.0]

    print("\n" + "#" * 100)
    print(context_label)
    print("#" * 100)

    res_anova = coherence_anova(subject=subject, freq=freq, joints=joints, kinematic_measure=kinematic_measure,
        acoustic_measure=acoustic_measure, lag=lag, compare=compare, average=average, value_type=value_type,
        transform=transform, )

    res_lmm = coherence_lmm(subject=subject, freq=freq, joints=joints, kinematic_measure=kinematic_measure,
                            acoustic_measure=acoustic_measure, lag=lag, compare=compare, average=average,
                            value_type=value_type, transform=transform, random_group=random_group,
                            re_formula=re_formula, reml=reml, run_lrt=run_lrt, continuous=continuous, )

    print_digestible_results(res_anova=res_anova, res_lmm=res_lmm, subject=subject, freq=freq, joints=joints,
        kinematic_measure=kinematic_measure, acoustic_measure=acoustic_measure, lag=lag, compare=compare,
        context_label=context_label, truncate=truncate, sort_key=sort_key, )

    if correct_pvalues:
        print_digestible_results_corrected(res_anova=res_anova, res_lmm=res_lmm, subject=subject, freq=freq,
            joints=joints, kinematic_measure=kinematic_measure, acoustic_measure=acoustic_measure, lag=lag,
            compare=compare, context_label=context_label, truncate=truncate, show_factors=False,
            alpha=correction_alpha, method=correction_method, format_output=format_output, sort_key=sort_key, )

    return res_anova, res_lmm


def plot_z_scores_lags(kinematic, n_boot=2000):
    ###########################################################################
    # 4b - Lag x acoustic measure plot (2x2: all joints / head / right hand / left hand)
    ###########################################################################
    # Mirrors the #2b frequency plot: same colors, same panel layout, same
    # legend-only-in-top-left / bold-title / Metropolis conventions, and the
    # same hardcoded value_type="z_scores" (independent of whatever kwargs
    # currently has, same reasoning as #2b: this plot is specifically about
    # z-scored coherence, regardless of what other sections are computing).
    #
    # Sign convention: the x-axis here is SIGN-FLIPPED from what the pipeline
    # computes internally, matching print_lag_ranking_table's convention and
    # the cross-correlation pipeline's convention -- NEGATIVE lag means
    # kinematics precedes acoustics. See print_lag_ranking_table's docstring
    # for how this was confirmed directly from analysis_functions.py.
    #
    # Peak dots + horizontal error bars: same convention as C04's cross-
    # correlation plot, but color-matched to each acoustic measure's curve
    # (rather than a single black overlay, since there's no single "mean
    # curve" here -- each line already IS a group-level result). The error
    # bar spans the 95% bootstrap CI on the peak's location (bootstrap_peak_lag,
    # same procedure/numbers as the results-section bootstrap table), NOT a
    # quadratic-fit interval -- kept consistent with the reporting decision
    # made for the results text, given these curves aren't reliably parabolic.
    # n_boot: lower this (e.g. 200-500) for faster draft iterations; use the
    # full 2000 for the final figure, matching the table's own N_BOOT.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ACOUSTIC_COLORS = {"envelope": "#6E3F7C", "intensity": "#028090", "pitch": "#758E4F", "f1": "#F6AE2D",
                       "f2": "#E67425"}
    ACOUSTIC_LABELS = {"envelope": "Envelope", "intensity": "Intensity", "pitch": "Pitch", "f1": "F1", "f2": "F2"}
    ACOUSTIC_ORDER = ["envelope", "intensity", "pitch", "f1", "f2"]  # legend order, matches the color list above

    # Panel order: top-left, top-right, bottom-left, bottom-right.
    JOINT_PANELS = {"All upper-body joints": list(UPPER_BODY_JOINTS), "Head": ["HeadFront"], "Right hand": ["HandInRight"],
                    "Left hand": ["HandInLeft"]}

    sns.set_style("whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False, "font.family": "Metropolis",
                                   "xtick.labelsize": 8, "ytick.labelsize": 8})

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    lag_values = [i / 1000 for i in range(-1000, 1000, 40)]

    i = 0

    for ax, (panel_title, panel_joints) in zip(axes, JOINT_PANELS.items()):
        for acoustic in ACOUSTIC_ORDER:
            res_acoustic = coherence_anova(subject=subjects, freq=freq, joints=panel_joints,
                                           kinematic_measure=["acceleration"], acoustic_measure=[acoustic], lag=lag_values,
                                           compare=("lags",), average="subject", value_type="z_scores", )
            print(f"FREQ: {freq}")

            means = res_acoustic["dataframe"].groupby("lag")["dv"].mean()
            means.index = -means.index  # sign flip -- negative = kinematics precedes acoustics
            means = means.sort_index()  # ascending, -1 s .. +1 s after the flip

            color = ACOUSTIC_COLORS[acoustic]
            ax.plot(means.index, means.values, color=color, label=ACOUSTIC_LABELS[acoustic], linewidth=1.8)

            # Peak dot + 95% bootstrap CI, reusing bootstrap_peak_lag so the
            # plotted peak/CI are numerically identical to the results table.
            peak_lag, ci_low, ci_high, _ = bootstrap_peak_lag(acoustic, panel_joints, panel_title,
                                                               kinematic_measure="acceleration", value_type="z_scores",
                                                               n_boot=n_boot)
            peak_val = means.loc[peak_lag] if peak_lag in means.index else means.iloc[
                (means.index - peak_lag).abs().argmin()]

            ax.errorbar(peak_lag, peak_val, xerr=[[peak_lag - ci_low], [ci_high - peak_lag]], fmt="none",
                       color=color, capsize=2, linewidth=0.5, zorder=4)
            ax.plot(peak_lag, peak_val, "o", color=color, markersize=5, zorder=5)

        ax.axhline(1.645, color="grey", linestyle="--", linewidth=1, alpha=0.7,
                   label="z = 1.645\n(p < .05,\none-tailed)" if panel_title == "All upper-body joints" else None)

        ax.set_title(panel_title, fontweight="bold", fontfamily="Metropolis")

        if i > 1:
            ax.set_xlabel("Lag (s) · Negative: kinematics lead acoustics", fontfamily="Metropolis")
        ax.set_ylabel("Coherence (z-score)", fontfamily="Metropolis")

        i += 1

    axes[0].legend(frameon=True, prop={"family": "Metropolis", "size": 8}, ncol=2, handlelength=1.2,
                   labelspacing=0.3, columnspacing=0.8)

    plt.tight_layout()

    path_png = op.join(PATH_PLOTS, "Coherence", "Audio", f"lag_by_acoustic_2x2_{kinematic}.png")
    os.makedirs(op.dirname(path_png), exist_ok=True)
    plt.savefig(path_png, dpi=300)
    plt.close(fig)

    print(f"Saved: {path_png}")

def plot_coherence_z_scores_frequency_bands(kinematic):
    ###########################################################################
    # 2b - Frequency x acoustic measure plot (2x2: all joints / head / right hand / left hand)
    ###########################################################################
    import matplotlib.pyplot as plt
    import seaborn as sns

    ACOUSTIC_COLORS = {"envelope": "#6E3F7C", "intensity": "#028090", "pitch": "#758E4F", "f1": "#F6AE2D", "f2": "#E67425"}
    ACOUSTIC_LABELS = {"envelope": "Envelope", "intensity": "Intensity", "pitch": "Pitch", "f1": "F1", "f2": "F2"}
    ACOUSTIC_ORDER = ["envelope", "intensity", "pitch", "f1", "f2"]  # legend order, matches the color list above

    # Panel order: top-left, top-right, bottom-left, bottom-right.
    JOINT_PANELS = {"All upper-body joints": list(UPPER_BODY_JOINTS), "Head": ["HeadFront"], "Right hand": ["HandInRight"],
                    "Left hand": ["HandInLeft"]}

    sns.set_style("whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False, "font.family": "Metropolis",
                                   "xtick.labelsize": 8, "ytick.labelsize": 8})

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    freq_points = [round(0.2 * i, 1) for i in range(2, 21)]

    for ax, (panel_title, panel_joints) in zip(axes, JOINT_PANELS.items()):
        for acoustic in ACOUSTIC_ORDER:
            res_acoustic = coherence_anova(subject=subjects, freq=freq_points, joints=panel_joints,
                                           kinematic_measure=[kinematic], acoustic_measure=[acoustic], lag=[0.0],
                                           compare=("freq",), average="subject", value_type="z_scores", )

            means = res_acoustic["dataframe"].groupby("frequency")["dv"].mean().sort_index()

            ax.plot(means.index, means.values, color=ACOUSTIC_COLORS[acoustic], label=ACOUSTIC_LABELS[acoustic],
                    linewidth=1.8)
        ax.axhline(1.645, color="grey", linestyle="--", linewidth=1, alpha=0.7,
                   label="z = 1.645\n(p < .05,\none-tailed)" if panel_title == "All upper-body joints" else None)

        ax.set_title(panel_title, fontweight="bold", fontfamily="Metropolis")
        ax.set_xlabel("Frequency (Hz)", fontfamily="Metropolis")
        ax.set_ylabel("Coherence (z-score)", fontfamily="Metropolis")

    axes[0].legend(frameon=True, prop={"family": "Metropolis", "size": 8}, ncol=2, handlelength=1.2,
                   labelspacing=0.3, columnspacing=0.8)

    plt.tight_layout()

    path_png = op.join(PATH_PLOTS, "Coherence", "Audio", f"frequency_by_acoustic_2x2_{kinematic}.png")
    os.makedirs(op.dirname(path_png), exist_ok=True)
    plt.savefig(path_png, dpi=300)
    plt.close(fig)

    print(f"Saved: {path_png}")

def bootstrap_peak_lag(acoustic, joints, joint_label, kinematic_measure="acceleration", value_type="z_scores",
                       n_boot=2000, seed=0):
    """
    Resamples subjects with replacement n_boot times; for each resample,
    computes the mean lag-coherence curve and records where its ACTUAL
    maximum lands (same empirical-peak convention as the plots -- no
    parabola fit, no shape assumption). The 2.5th/97.5th percentiles of
    that distribution give a 95% CI on the peak's location.

    p-value: two-tailed bootstrap p, p = 2 * min(P(peak <= 0), P(peak >= 0))
    capped at 1 -- lets this go through the same BH correction as every
    other table in this pipeline, rather than just a binary CI-excludes-0
    flag.
    """
    lag_values = [i / 1000 for i in range(-1000, 1000, 40)]
    df = build_coherence_tidy_dataframe(subject=subjects, freq=freq, joints=joints,
                                        kinematic_measure=[kinematic_measure], acoustic_measure=[acoustic],
                                        lag=lag_values, average=kwargs.get("average", "subject"),
                                        value_type=value_type, )
    df = df.copy()
    df["dv"] = df["coherence"]
    df = df.dropna(subset=["dv"])
    df = df.groupby(["subject", "lag"], observed=True)["dv"].mean().reset_index()
    df["lag_display"] = -df["lag"]

    subj_list = sorted(df["subject"].unique())
    n_subj = len(subj_list)

    pivot = df.pivot(index="subject", columns="lag_display", values="dv").loc[subj_list]
    lag_grid = pivot.columns.to_numpy()
    data_matrix = pivot.to_numpy()  # shape (n_subj, n_lags)

    rng = np.random.default_rng(seed)
    boot_peaks = np.empty(n_boot)

    for b in range(n_boot):
        sample_idx = rng.integers(0, n_subj, size=n_subj)
        resampled_mean = data_matrix[sample_idx].mean(axis=0)
        boot_peaks[b] = lag_grid[np.argmax(resampled_mean)]

    ci_low, ci_high = np.percentile(boot_peaks, [2.5, 97.5])
    observed_peak = lag_grid[np.argmax(data_matrix.mean(axis=0))]

    p_below = np.mean(boot_peaks <= 0)
    p_above = np.mean(boot_peaks >= 0)
    p_value = min(1.0, 2 * min(p_below, p_above))

    print(f"Bootstrap peak-lag CI — {kinematic_measure} × {acoustic} — {joint_label}: "
          f"peak = {observed_peak:.3f}, 95% CI = [{ci_low:.3f}, {ci_high:.3f}], p = {p_value:.4f}")

    return observed_peak, ci_low, ci_high, p_value


if __name__ == "__main__":
    run_lrt = True
    # value_type: averages or z_scores
    kwargs = {"truncate": 30, "value_type": "z_scores", "average": "subject", "format_output": "tabs"}

    subjects = SUBJECTS_NO_OUTLIERS
    joints = ["HeadFront", "HandInRight", "HandInLeft"]

    ###########################################################################
    # 1 - Which combination of measures shows the highest coherence?
    ###########################################################################
    freq = [(0.4, 2)]

    run_anova_lmm_digestible(context_label="Measure combinations — all upper-body joints",
                             compare=("kinematic_measures", "acoustic_measures"), subject=subjects,
                             joints=list(UPPER_BODY_JOINTS),
                             kinematic_measure=["velocity", "acceleration", "jerk"], acoustic_measure=DERIVATIVES,
                             freq=freq, lag=[0.0], run_lrt=run_lrt, **kwargs)

    run_anova_lmm_digestible(context_label="Measure combinations — right hand",
                             compare=("kinematic_measures", "acoustic_measures"), subject=subjects,
                             joints=["HandInRight"], kinematic_measure=["velocity", "acceleration", "jerk"],
                             acoustic_measure=DERIVATIVES, freq=freq, lag=[0.0], run_lrt=run_lrt, **kwargs)

    run_anova_lmm_digestible(context_label="Measure combinations — left hand",
                             compare=("kinematic_measures", "acoustic_measures"), subject=subjects,
                             joints=["HandInLeft"], kinematic_measure=["velocity", "acceleration", "jerk"],
                             acoustic_measure=DERIVATIVES, freq=freq, lag=[0.0], run_lrt=run_lrt, **kwargs)

    run_anova_lmm_digestible(context_label="Measure combinations — head",
                             compare=("kinematic_measures", "acoustic_measures"), subject=subjects,
                             joints=["HeadFront"], kinematic_measure=["velocity", "acceleration", "jerk"],
                             acoustic_measure=DERIVATIVES, freq=freq, lag=[0.0], run_lrt=run_lrt, **kwargs)

    ###########################################################################
    # 2 - Which frequency bands cohere best for each acoustic measure?
    ###########################################################################
    # freq_bands = [(round(i / 10, 1), round(i / 10 + 0.2, 1)) for i in range(4, 40, 2)]
    # freq_bands = [(0.2, 0.4), (0.4, 0.8), (0.8, 1.6), (1.6, 3.2)]
    #
    # freq_points = [round(0.2 * i, 1) for i in range(2, 21)]  # 0.2, 0.4, ..., 4.0 -- kept active, #2b (plot) needs it
    # kinematic = "acceleration"
    #
    # def freq_sort_key(summary):
    #     """Re-sort a frequency-effect summary ascending by frequency (0.4 Hz -> 4.0 Hz) instead of by coherence."""
    #     summary = summary.copy()
    #     summary["_freq_numeric"] = summary["freq_label"].str.replace(" Hz", "", regex=False).astype(float)
    #     return summary.sort_values("_freq_numeric", ascending=True)
    #
    # for acoustic in ["envelope", "intensity", "pitch", "f1", "f2"]:
    #     run_anova_lmm_digestible(context_label=f"Frequency ranking — {kinematic} × {acoustic}", compare=("freq",),
    #                              subject=subjects, joints=list(UPPER_BODY_JOINTS), kinematic_measure=[kinematic],
    #                              acoustic_measure=[acoustic], freq=freq_points, lag=[0.0], run_lrt=run_lrt,
    #                              sort_key=freq_sort_key, **kwargs)
    #
    #     run_anova_lmm_digestible(context_label=f"Frequency ranking — {kinematic} × {acoustic} — right hand",
    #                              compare=("freq",), subject=subjects, joints=["HandInRight"],
    #                              kinematic_measure=[kinematic], acoustic_measure=[acoustic], freq=freq_points,
    #                              lag=[0.0], run_lrt=run_lrt, sort_key=freq_sort_key, **kwargs)
    #
    #     run_anova_lmm_digestible(context_label=f"Frequency ranking — {kinematic} × {acoustic} — left hand",
    #                              compare=("freq",), subject=subjects, joints=["HandInLeft"],
    #                              kinematic_measure=[kinematic], acoustic_measure=[acoustic], freq=freq_points,
    #                              lag=[0.0], run_lrt=run_lrt, sort_key=freq_sort_key, **kwargs)
    #
    #     run_anova_lmm_digestible(context_label=f"Frequency ranking — {kinematic} × {acoustic} — head",
    #                              compare=("freq",), subject=subjects, joints=["HeadFront"],
    #                              kinematic_measure=[kinematic], acoustic_measure=[acoustic], freq=freq_points,
    #                              lag=[0.0], run_lrt=run_lrt, sort_key=freq_sort_key, **kwargs)

    # run_anova_lmm_digestible(context_label="Frequency × acoustic measure — all upper-body joints",
    #                          compare=("freq", "acoustic_measures"), subject=subjects,
    #                          joints=list(UPPER_BODY_JOINTS_REREF), kinematic_measure=[kinematic],
    #                          acoustic_measure=DERIVATIVES, freq=freq_bands, lag=[0.0], run_lrt=run_lrt, **kwargs)
    #
    # run_anova_lmm_digestible(context_label="Frequency × acoustic measure — right hand",
    #                          compare=("freq", "acoustic_measures"), subject=subjects, joints=["HandRight"],
    #                          kinematic_measure=[kinematic], acoustic_measure=DERIVATIVES, freq=freq_bands, lag=[0.0],
    #                          run_lrt=run_lrt, **kwargs)



    ###########################################################################
    # 3 - Which joints have the highest coherence?
    ###########################################################################
    # SpineMid first so it's the LMM reference level (see coherence_lmm: the
    # reference for each factor is simply the first element of the list you
    # pass, not alphabetical or data-driven) -- a proximal joint is a more
    # interpretable baseline than whatever happened to sort first.
    # joints_spinemid_ref = ["Chest"] + [j for j in UPPER_BODY_JOINTS if j != "Chest"]
    # kinematic = "acceleration"
    # freq = [0.6]
    #
    # for acoustic in ["envelope", "intensity", "pitch", "f1", "f2"]:
    #     run_anova_lmm_digestible(context_label=f"Joint ranking — {kinematic} × {acoustic}", compare=("joints",),
    #                              subject=subjects, joints=joints_spinemid_ref, kinematic_measure=[kinematic],
    #                              acoustic_measure=[acoustic], freq=freq, lag=[0.0], run_lrt=run_lrt, **kwargs)

    ###########################################################################
    # 4 - At what lags can we find the highest coherence?
    ###########################################################################
    # lag is continuous (see coherence_lmm(..., continuous={"lags": 2})), so each
    # call's own within-table correction is a no-op (single omnibus test, no
    # per-level dummy coefficients -- see the "why no LMM coef." discussion).
    # The real multiple-comparisons question here is across the 5 acoustic
    # measures within a joint-set, so that's corrected separately below,
    # one family per joint-set (same "one family per group" convention already
    # used for the cross-correlation pipeline).
    lag_values = [i / 1000 for i in range(-1000, 1000, 40)]
    freq = [0.6]
    kinematic = "acceleration"

    # def print_lag_ranking_table(res_anova, res_lmm, context_label, value_decimals=3, format_output="console"):
    #     """
    #     Custom table for lag-ranking results ONLY -- not used by any other
    #     section. Two differences from the standard digestible table:
    #
    #     1. Rows are ordered by lag value itself (-1 s to +1 s), not by
    #        coherence, since for a continuous predictor there's no per-level
    #        coefficient to rank by in the first place (see the "why is LMM
    #        coef. blank" discussion) and a chronological order is more
    #        readable here than an arbitrary one.
    #
    #     2. The displayed lag has its SIGN FLIPPED relative to what the
    #        pipeline computes internally. Confirmed directly from
    #        analysis_functions.py's array indexing: internally, a POSITIVE
    #        lag means kinematics precedes acoustics. This flips that so the
    #        displayed sign instead matches the cross-correlation pipeline's
    #        convention (negative = kinematics precedes), so the two lag
    #        conventions read consistently if placed side by side in the
    #        thesis. If that other convention ever changes, this flip needs
    #        to be revisited too.
    #
    #     The single highest-coherence lag (by marginal mean -- the sign flip
    #     doesn't change which lag has the highest mean, only its printed
    #     label) is bolded in console mode. In tabs mode, bold can't survive a
    #     plain-text paste into Excel, so the peak is instead marked with
    #     "Yes" in an explicit third column -- filterable/sortable, and won't
    #     get lost on paste.
    #     """
    #
    #     anova_p = _get_anova_effect_p(res_anova, ("lag",))
    #     lmm_p, _ = _get_lmm_effect_p_for_digest(res_lmm, ("lag",))
    #
    #     means = res_anova["dataframe"].groupby("lag")["dv"].mean()
    #     means.index = -means.index  # sign flip -- see docstring
    #     means = means.sort_index()  # ascending, -1 .. +1 after the flip
    #
    #     peak_lag = means.idxmax()
    #
    #     if format_output == "tabs":
    #         print(context_label)
    #         print("\t".join(["", "ANOVA", "LMM"]))
    #         print("\t".join(["Omnibus", _format_p_tabs(anova_p), _format_p_tabs(lmm_p)]))
    #         print()
    #         print("\t".join(["Lag (s)", "Coherence", "Peak"]))
    #
    #         for lag_value, mean_value in means.items():
    #             print("\t".join([f"{lag_value:.2f}", f"{mean_value:.{value_decimals}f}",
    #                              "Yes" if lag_value == peak_lag else "", ]))
    #
    #         print()
    #         return
    #
    #     print(f"\n{context_label}")
    #     print(f"Omnibus (quadratic lag effect): ANOVA {_p_digestible_colored(anova_p)}, "
    #           f"LMM {_p_digestible_colored(lmm_p)}")
    #     print()
    #
    #     header = f"{'Lag (s)':>10} | {'Coherence':>12}"
    #     print(header)
    #     print("-" * len(header))
    #
    #     for lag_value, mean_value in means.items():
    #         row = f"{lag_value:>10.2f} | {mean_value:>{12}.{value_decimals}f}"
    #         if lag_value == peak_lag:
    #             print(f"\033[1m{row}  <-- peak\033[0m")
    #         else:
    #             print(row)
    #
    # for joint_set_label, joint_set in {"all upper-body joints": list(UPPER_BODY_JOINTS),
    #                                    "right hand": ["HandInRight"], "left hand": ["HandInLeft"],
    #                                    "head": ["HeadFront"], }.items():
    #     anova_ps, lmm_ps, acoustic_names = [], [], []
    #
    #     for acoustic in ["envelope", "intensity", "pitch", "f1", "f2"]:
    #         res_anova = coherence_anova(subject=subjects, freq=freq, joints=joint_set, kinematic_measure=[kinematic],
    #                                     acoustic_measure=[acoustic], lag=lag_values, compare=("lags",),
    #                                     average=kwargs.get("average", "subject"),
    #                                     value_type=kwargs.get("value_type", "averages"), )
    #         res_lmm = coherence_lmm(subject=subjects, freq=freq, joints=joint_set, kinematic_measure=[kinematic],
    #                                 acoustic_measure=[acoustic], lag=lag_values, compare=("lags",),
    #                                 average=kwargs.get("average", "subject"),
    #                                 value_type=kwargs.get("value_type", "averages"), random_group="subject",
    #                                 re_formula="1", reml=False, run_lrt=run_lrt, continuous={"lags": 2}, )
    #
    #         print_lag_ranking_table(res_anova, res_lmm,
    #                                 context_label=f"Lag ranking — {kinematic} × {acoustic} — {joint_set_label}",
    #                                 format_output=kwargs.get("format_output", "console"), )
    #
    #         anova_ps.append(_get_anova_effect_p(res_anova, ("lag",)))
    #         lmm_p, _ = _get_lmm_effect_p_for_digest(res_lmm, ("lag",))
    #         lmm_ps.append(lmm_p)
    #         acoustic_names.append(acoustic)
    #
    #         print("Is the peak lag significantly different from zero?")
    #         print(res_lmm["model"].pvalues["lag"])
    #
    #         print("What is the peak lag?")
    #         b = res_lmm["model"].params["lag"]  # linear coefficient
    #         a = res_lmm["model"].params["I(lag ** 2)"]  # quadratic coefficient -- adjust the key if it prints differently above
    #
    #         peak_lag_display = b / (2 * a)  # already sign-flipped to match your negative=kinematics-precedes convention
    #         print(peak_lag_display)
    #
    #     anova_ps_corr = _bh_correct(anova_ps)
    #     lmm_ps_corr = _bh_correct(lmm_ps)
    #
    #     if kwargs.get("format_output", "console") == "tabs":
    #         print(f"Lag effect across acoustic measures — {joint_set_label} "
    #               f"(BH-corrected, family = these 5 acoustic measures)")
    #         print("\t".join(["", "Uncorrected", "", "p-corrected", ""]))
    #         print("\t".join(["Acoustic measure", "ANOVA", "LMM", "ANOVA", "LMM"]))
    #
    #         for name, ap, lp, apc, lpc in zip(acoustic_names, anova_ps, lmm_ps, anova_ps_corr, lmm_ps_corr):
    #             print("\t".join([name, _format_p_tabs(ap), _format_p_tabs(lp), _format_p_tabs(apc),
    #                              _format_p_tabs(lpc), ]))
    #         print()
    #     else:
    #         print(f"\n---- Lag effect across acoustic measures — {joint_set_label} "
    #               f"(BH-corrected, family = these 5 acoustic measures) ----")
    #         print(f"{'Acoustic measure':<16} | {'ANOVA':>10} | {'LMM':>10} | {'ANOVA (p-corr)':>16} | "
    #               f"{'LMM (p-corr)':>14}")
    #
    #         for name, ap, lp, apc, lpc in zip(acoustic_names, anova_ps, lmm_ps, anova_ps_corr, lmm_ps_corr):
    #             print(f"{name:<16} | {_p_digestible_colored(ap):>10} | {_p_digestible_colored(lp):>10} | "
    #                   f"{_p_digestible_colored(apc):>16} | {_p_digestible_colored(lpc):>14}")
    # #
    # # kinematic = "velocity"
    # # plot_coherence_z_scores_frequency_bands()
    # # plot_z_scores_lags()
    #
    # anova_ps_corr = _bh_correct(anova_ps)
    # lmm_ps_corr = _bh_correct(lmm_ps)
    #
    # if kwargs.get("format_output", "console") == "tabs":
    #     print(f"Lag effect across acoustic measures — {joint_set_label} "
    #           f"(BH-corrected, family = these 5 acoustic measures)")
    #     print("\t".join(["", "Uncorrected", "", "p-corrected", ""]))
    #     print("\t".join(["Acoustic measure", "ANOVA", "LMM", "ANOVA", "LMM"]))
    #
    #     for name, ap, lp, apc, lpc in zip(acoustic_names, anova_ps, lmm_ps, anova_ps_corr, lmm_ps_corr):
    #         print("\t".join([name, _format_p_tabs(ap), _format_p_tabs(lp), _format_p_tabs(apc),
    #                          _format_p_tabs(lpc), ]))
    #     print()
    # else:
    #     print(f"\n---- Lag effect across acoustic measures — {joint_set_label} "
    #           f"(BH-corrected, family = these 5 acoustic measures) ----")
    #     print(f"{'Acoustic measure':<16} | {'ANOVA':>10} | {'LMM':>10} | {'ANOVA (p-corr)':>16} | "
    #           f"{'LMM (p-corr)':>14}")
    #
    #     for name, ap, lp, apc, lpc in zip(acoustic_names, anova_ps, lmm_ps, anova_ps_corr, lmm_ps_corr):
    #         print(f"{name:<16} | {_p_digestible_colored(ap):>10} | {_p_digestible_colored(lp):>10} | "
    #               f"{_p_digestible_colored(apc):>16} | {_p_digestible_colored(lpc):>14}")

    ###########################################################################
    # 6 - Does the lag-coherence curve differ between head and hand?
    ###########################################################################
    # Set these two per dataset before running:
    #   Kinect:   HEAD_JOINT = "Head",      HAND_JOINT = "HandRight"
    #   Qualisys: HEAD_JOINT = "HeadFront", HAND_JOINT = "HandInRight"
    # HEAD_JOINT = "HeadFront"
    # HAND_JOINT = "HandInLeft"
    #
    # kinematic = "acceleration"
    #
    # JOINT_GROUP_MAP = {HEAD_JOINT: "Head", HAND_JOINT: "Hand"}
    #
    #
    # def test_head_vs_hand_lag(acoustic, subjects, freq, lag_values, kinematic_measure=kinematic):
    #     """
    #     Formal test of whether the lag-coherence curve's shape (and therefore its
    #     peak) differs between the head and hand, via a joint_group x lag
    #     interaction. Model:
    #
    #         dv ~ lag + I(lag**2) + C(joint_group)
    #              + lag:C(joint_group) + I(lag**2):C(joint_group)
    #
    #     (+ C(subject) as a fixed effect for the ANOVA version; random intercept
    #     on subject for the LMM version.) Tested by comparing this full model
    #     against the same model with BOTH interaction terms dropped -- an F-test
    #     for ANOVA, a likelihood-ratio test (2 df) for LMM. A significant result
    #     means the two groups' peak lags are statistically distinguishable, not
    #     just visually different in a table.
    #     """
    #     df = build_coherence_tidy_dataframe(subject=subjects, freq=freq, joints=[HEAD_JOINT, HAND_JOINT],
    #                                         kinematic_measure=[kinematic_measure], acoustic_measure=[acoustic],
    #                                         lag=lag_values, average=kwargs.get("average", "subject"),
    #                                         value_type=kwargs.get("value_type", "averages"), )
    #     df = df.copy()
    #
    #     # Replicate what coherence_anova()/coherence_lmm() do internally: raw output has
    #     # a "coherence" column, not "dv" -- rename, drop NaNs, then aggregate down to one
    #     # row per (subject, joint_group, lag), same as their own group_cols/agg step.
    #     df["dv"] = df["coherence"]
    #     df = df.dropna(subset=["dv"])
    #     df["joint_group"] = df["joint"].map(JOINT_GROUP_MAP)
    #     df = df.groupby(["subject", "joint_group", "lag"], observed=True)["dv"].mean().reset_index()
    #
    #     # This is inherently a WITHIN-subject comparison (does a given person's head lag
    #     # differ from their own hand lag) -- a subject present in only one joint_group
    #     # contributes nothing to that contrast but still consumes a C(subject) fixed-effect
    #     # parameter, which can distort the interaction test. Restrict to subjects present
    #     # in BOTH groups before fitting.
    #     subjects_per_group = df.groupby("joint_group")["subject"].agg(set)
    #     common_subjects = set.intersection(*subjects_per_group)
    #     dropped = set.union(*subjects_per_group) - common_subjects
    #
    #     if dropped:
    #         print(f"  [NOTE] Dropping {len(dropped)} subject(s) missing from one joint_group for {acoustic}: "
    #               f"{sorted(dropped)}")
    #         df = df[df["subject"].isin(common_subjects)]
    #
    #     full_formula = "dv ~ C(subject) + lag + I(lag**2) + C(joint_group) + lag:C(joint_group) + I(lag**2):C(joint_group)"
    #     reduced_formula = "dv ~ C(subject) + lag + I(lag**2) + C(joint_group)"
    #
    #     full_ols = ols(full_formula, data=df).fit()
    #     reduced_ols = ols(reduced_formula, data=df).fit()
    #     anova_test = anova_lm(reduced_ols, full_ols)
    #     anova_p = anova_test["Pr(>F)"].iloc[1]
    #
    #     full_lmm_formula = "dv ~ lag + I(lag**2) + C(joint_group) + lag:C(joint_group) + I(lag**2):C(joint_group)"
    #     reduced_lmm_formula = "dv ~ lag + I(lag**2) + C(joint_group)"
    #
    #     full_lmm = mixedlm(full_lmm_formula, data=df, groups=df["subject"]).fit(reml=False)
    #     reduced_lmm = mixedlm(reduced_lmm_formula, data=df, groups=df["subject"]).fit(reml=False)
    #
    #     if np.isfinite(full_lmm.llf) and np.isfinite(reduced_lmm.llf):
    #         lr_stat = 2 * (full_lmm.llf - reduced_lmm.llf)
    #         lmm_p = chi2.sf(lr_stat, df=2) if lr_stat >= 0 else np.nan
    #     else:
    #         lmm_p = np.nan
    #
    #     print(f"Head vs hand lag-curve interaction — {kinematic} × {acoustic}: "
    #           f"ANOVA p = {anova_p:.4f}, LMM (LRT) p = {lmm_p if np.isnan(lmm_p) else f'{lmm_p:.4f}'}")
    #
    #     return anova_p, lmm_p
    #
    #
    # anova_ps, lmm_ps, acoustic_names = [], [], []
    #
    # for acoustic in ["envelope", "intensity", "pitch", "f1", "f2"]:
    #     ap, lp = test_head_vs_hand_lag(acoustic, subjects, freq, lag_values)
    #     anova_ps.append(ap)
    #     lmm_ps.append(lp)
    #     acoustic_names.append(acoustic)
    #
    # anova_ps_corr = _bh_correct(anova_ps)
    # lmm_ps_corr = _bh_correct(lmm_ps)
    #
    # if kwargs.get("format_output", "console") == "tabs":
    #     print(f"Head vs hand lag interaction across acoustic measures "
    #           f"(BH-corrected, family = these 5 acoustic measures)")
    #     print("\t".join(["", "Uncorrected", "", "p-corrected", ""]))
    #     print("\t".join(["Acoustic measure", "ANOVA", "LMM", "ANOVA", "LMM"]))
    #
    #     for name, ap, lp, apc, lpc in zip(acoustic_names, anova_ps, lmm_ps, anova_ps_corr, lmm_ps_corr):
    #         print("\t".join([name, _format_p_tabs(ap), _format_p_tabs(lp), _format_p_tabs(apc), _format_p_tabs(lpc), ]))
    #     print()
    # else:
    #     print(f"\n---- Head vs hand lag interaction across acoustic measures (BH-corrected, family = these 5) ----")
    #     print(f"{'Acoustic measure':<16} | {'ANOVA':>10} | {'LMM':>10} | {'ANOVA (p-corr)':>16} | {'LMM (p-corr)':>14}")
    #     for name, ap, lp, apc, lpc in zip(acoustic_names, anova_ps, lmm_ps, anova_ps_corr, lmm_ps_corr):
    #         print(f"{name:<16} | {_p_digestible_colored(ap):>10} | {_p_digestible_colored(lp):>10} | "
    #               f"{_p_digestible_colored(apc):>16} | {_p_digestible_colored(lpc):>14}")

    # ###########################################################################
    # # 7 - Do subjects vary in their OWN lag-coherence slope (not just baseline)?
    # ###########################################################################
    # # Set per dataset/joint-set of interest:
    # #   Kinect:   RS_JOINTS = list(UPPER_BODY_JOINTS_REREF)  (or ["HandRight"], ["Head"], etc.)
    # #   Qualisys: RS_JOINTS = list(UPPER_BODY_JOINTS_REREF)  (or ["HandInRight"], ["HeadFront"], etc.)
    # RS_JOINTS = list(UPPER_BODY_JOINTS)
    # RS_JOINT_LABEL = "all upper-body joints"
    # kinematic = "acceleration"
    #
    # def test_random_slope_lag(acoustic, kinematic_measure=kinematic):
    #     """
    #     Tests whether subjects differ in their own lag-coherence SLOPE, not just
    #     their baseline level -- i.e. does the random-intercept-only model
    #     (re_formula="1", what every other LMM in this pipeline uses) understate
    #     real between-subject variability in the effect itself?
    #
    #     Reduced model: random intercept only    -- re_formula="1"
    #     Full model:    random intercept + slope -- re_formula="1 + lag"
    #     (only the LINEAR lag term gets a random slope; adding lag**2 too would
    #     very likely fail to converge given the boundary warnings already seen
    #     with random-intercept-only models.)
    #
    #     Compared via LRT (2 df: slope variance + intercept-slope covariance --
    #     note the same boundary-asymptotics caveat already flagged elsewhere in
    #     this pipeline applies here, more so, since variance-covariance
    #     components are exactly where that caveat bites hardest).
    #
    #     Reports the full model's estimated variance components directly:
    #     between-subject SD of the intercept, between-subject SD of the slope,
    #     and their correlation -- these are the numbers for the subject-
    #     variability discussion, independent of whether the LRT itself reaches
    #     significance.
    #     """
    #     res_lmm_reduced = coherence_lmm(subject=subjects, freq=freq, joints=RS_JOINTS,
    #                                     kinematic_measure=[kinematic_measure], acoustic_measure=[acoustic],
    #                                     lag=lag_values, compare=("lags",), average=kwargs.get("average", "subject"),
    #                                     value_type=kwargs.get("value_type", "averages"), random_group="subject",
    #                                     re_formula="1", reml=False, run_lrt=False, continuous={"lags": 2}, )
    #
    #     # Uncorrelated (variance-components) random slope instead of a jointly-correlated
    #     # one: re_formula stays "1" (random intercept only), and vc_formula adds an
    #     # INDEPENDENT random slope for lag. This is a more parsimonious, better-
    #     # conditioned structure for small N -- a jointly-correlated "1 + lag" random
    #     # effect asks the model to also estimate the intercept-slope covariance, which
    #     # is very likely unidentifiable at N=6 and tends to show up as a spuriously
    #     # near-1 correlation (exactly what an initial run of this test showed).
    #     res_lmm_full = coherence_lmm(subject=subjects, freq=freq, joints=RS_JOINTS,
    #                                  kinematic_measure=[kinematic_measure], acoustic_measure=[acoustic],
    #                                  lag=lag_values, compare=("lags",), average=kwargs.get("average", "subject"),
    #                                  value_type=kwargs.get("value_type", "averages"), random_group="subject",
    #                                  re_formula="1", vc_formula={"lag_slope": "0 + lag"}, reml=False, run_lrt=False,
    #                                  continuous={"lags": 2}, )
    #
    #     reduced_model = res_lmm_reduced["model"]
    #     full_model = res_lmm_full["model"]
    #
    #     # Stricter convergence check: .converged alone can read True even after a
    #     # documented optimizer failure earlier in a multi-method fallback sequence
    #     # (seen directly in the console log for one acoustic measure last run) --
    #     # cross-check against mle_retvals, including the gradient norm at the optimum.
    #     retvals = getattr(full_model, "mle_retvals", {}) or {}
    #     grad_norm = retvals.get("gnorm", retvals.get("|grad|", None))
    #     converged = bool(getattr(full_model, "converged", False)) and bool(retvals.get("converged", True)) and (
    #         grad_norm is None or grad_norm < 1.0)
    #     if not converged:
    #         print(f"  [WARNING] Random-slope model did NOT reliably converge for {acoustic} "
    #               f"(mle_retvals={retvals}) -- treat these numbers cautiously.")
    #
    #     if np.isfinite(full_model.llf) and np.isfinite(reduced_model.llf):
    #         lr_stat = 2 * (full_model.llf - reduced_model.llf)
    #         lrt_p = chi2.sf(lr_stat, df=1) if lr_stat >= 0 else np.nan  # 1 df: one variance component added
    #     else:
    #         lrt_p = np.nan
    #
    #     intercept_var = full_model.cov_re.iloc[0, 0] if hasattr(full_model.cov_re, "iloc") else full_model.cov_re[
    #         0, 0]
    #     slope_var = full_model.vcomp[0] if hasattr(full_model, "vcomp") and len(full_model.vcomp) else np.nan
    #     corr_is = np.nan  # not estimated under this independent structure -- see note above
    #
    #     # ICC: fraction of total variance attributable to consistent between-subject
    #     # differences, vs. within-subject/residual noise. Computed from the REDUCED
    #     # (random-intercept-only) model -- the same model every other LMM in this
    #     # pipeline already fits, reused here for free rather than fitting anything new.
    #     reduced_subject_var = reduced_model.cov_re.iloc[0, 0] if hasattr(reduced_model.cov_re, "iloc") else \
    #         reduced_model.cov_re[0, 0]
    #     reduced_residual_var = reduced_model.scale
    #     icc = reduced_subject_var / (reduced_subject_var + reduced_residual_var)
    #
    #     print(f"Random-slope test — {kinematic_measure} × {acoustic} — {RS_JOINT_LABEL}: "
    #           f"LRT p = {lrt_p if np.isnan(lrt_p) else f'{lrt_p:.4f}'}, "
    #           f"subject SD(intercept) = {np.sqrt(intercept_var):.4f}, "
    #           f"subject SD(slope) = {np.sqrt(slope_var):.4f}, "
    #           f"ICC = {icc:.4f}, "
    #           f"converged = {converged}")
    #
    #     return lrt_p, np.sqrt(intercept_var), np.sqrt(slope_var), corr_is, icc, converged
    #
    #
    # lrt_ps, sd_intercepts, sd_slopes, corrs, iccs, converged_flags, acoustic_names = [], [], [], [], [], [], []
    #
    # for acoustic in ["envelope", "intensity", "pitch", "f1", "f2"]:
    #     lp, sdi, sds, corr, icc, conv = test_random_slope_lag(acoustic)
    #     lrt_ps.append(lp)
    #     sd_intercepts.append(sdi)
    #     sd_slopes.append(sds)
    #     corrs.append(corr)
    #     iccs.append(icc)
    #     converged_flags.append(conv)
    #     acoustic_names.append(acoustic)
    #
    # lrt_ps_corr = _bh_correct(lrt_ps)
    #
    # if kwargs.get("format_output", "console") == "tabs":
    #     print(f"Random-slope test across acoustic measures — {RS_JOINT_LABEL} "
    #           f"(BH-corrected LRT, family = these 5 acoustic measures)")
    #     print("\t".join(["Acoustic measure", "LRT p", "LRT p (p-corr)", "SD(intercept)", "SD(slope)",
    #                      "corr(int,slope)", "ICC", "Converged"]))
    #     for name, lp, lpc, sdi, sds, corr, icc, conv in zip(acoustic_names, lrt_ps, lrt_ps_corr, sd_intercepts,
    #                                                          sd_slopes, corrs, iccs, converged_flags):
    #         print("\t".join([name, _format_p_tabs(lp), _format_p_tabs(lpc), f"{sdi:.4f}", f"{sds:.4f}",
    #                          "" if np.isnan(corr) else f"{corr:.3f}", f"{icc:.4f}", str(conv), ]))
    #     print()
    # else:
    #     print(f"\n---- Random-slope test across acoustic measures — {RS_JOINT_LABEL} "
    #           f"(BH-corrected LRT, family = these 5) ----")
    #     print(f"{'Acoustic measure':<16} | {'LRT p':>10} | {'LRT p-corr':>12} | {'SD(int.)':>10} | "
    #           f"{'SD(slope)':>10} | {'corr':>8} | {'ICC':>8} | {'Conv.':>6}")
    #     for name, lp, lpc, sdi, sds, corr, icc, conv in zip(acoustic_names, lrt_ps, lrt_ps_corr, sd_intercepts,
    #                                                          sd_slopes, corrs, iccs, converged_flags):
    #         print(f"{name:<16} | {_p_digestible_colored(lp):>10} | {_p_digestible_colored(lpc):>12} | "
    #               f"{sdi:>10.4f} | {sds:>10.4f} | "
    #               f"{'—' if np.isnan(corr) else f'{corr:.3f}':>8} | {icc:>8.4f} | {str(conv):>6}")

    ###########################################################################
    # 9 - Per-subject coherence vs. lag, 2x2 (all joints / head / right hand / left hand)
    ###########################################################################
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Panel order: top-left, top-right, bottom-left, bottom-right (same convention as #2b/#4b).
    # Set per dataset:
    #   Kinect:   JOINT_PANELS = {"All upper-body joints": list(UPPER_BODY_JOINTS_REREF), "Head": ["Head"],
    #                             "Right hand": ["HandRight"], "Left hand": ["HandLeft"]}
    #   Qualisys: JOINT_PANELS = {"All upper-body joints": list(UPPER_BODY_JOINTS_REREF), "Head": ["HeadFront"],
    #                             "Right hand": ["HandInRight"], "Left hand": ["HandInLeft"]}
    JOINT_PANELS = {"All upper-body joints": list(UPPER_BODY_JOINTS), "Head": ["HeadFront"],
                    "Right hand": ["HandInRight"], "Left hand": ["HandInLeft"]}
    kinematic = "acceleration"

    def _draw_subject_lag_panel(ax, acoustic, joints, joint_label, value_type, kinematic_measure,
                                peak_search_range, show_legend):
        """
        Draws one panel: one line per subject (coherence vs. lag), a small dot
        at each subject's own peak, and a bold mean curve with an SE-of-peak-lag
        errorbar at its own peak -- same conventions as C04's _draw_subpanel().

        Sign convention: build_coherence_tidy_dataframe's "lag" column follows
        the pipeline's INTERNAL convention (positive = kinematics precedes --
        see print_lag_ranking_table's docstring). Negated here before plotting,
        same as everywhere else in this script, so negative = kinematics
        precedes acoustics.
        """
        df = build_coherence_tidy_dataframe(subject=subjects, freq=freq, joints=joints,
                                            kinematic_measure=[kinematic_measure], acoustic_measure=[acoustic],
                                            lag=lag_values, average=kwargs.get("average", "subject"),
                                            value_type=value_type, )
        df = df.copy()
        df["dv"] = df["coherence"]
        df = df.dropna(subset=["dv"])
        df = df.groupby(["subject", "lag"], observed=True)["dv"].mean().reset_index()
        df["lag_display"] = -df["lag"]

        subj_list = sorted(df["subject"].unique())

        def peak_idx(lag_vals, curve_vals):
            if peak_search_range is not None:
                mask = np.abs(lag_vals) <= peak_search_range
                if not mask.any():
                    raise ValueError(f"No lags found within peak_search_range={peak_search_range}.")
                candidates = np.where(mask)[0]
                return candidates[np.argmax(curve_vals[mask])]
            return int(np.argmax(curve_vals))

        subject_curves = []
        for subj in subj_list:
            sub_df = df[df["subject"] == subj].sort_values("lag_display")
            lag_vals = sub_df["lag_display"].to_numpy()
            curve_vals = sub_df["dv"].to_numpy()
            subject_curves.append((lag_vals, curve_vals))

            color = SUBJECT_COLORS.get(subj, "grey") if "SUBJECT_COLORS" in globals() else "grey"
            label = SUBJECT_ANON.get(subj, subj) if "SUBJECT_ANON" in globals() else subj

            ax.plot(lag_vals, curve_vals, color=color, linewidth=1.2, alpha=0.8, label=label, zorder=3)
            pidx = peak_idx(lag_vals, curve_vals)
            ax.plot(lag_vals[pidx], curve_vals[pidx], "o", color=color, markersize=4, zorder=4)

        common_lags = subject_curves[0][0]
        mean_curve = np.mean([c for _, c in subject_curves], axis=0)
        ax.plot(common_lags, mean_curve, color="black", linewidth=2.2, label="Mean", zorder=5)

        peak_lags = np.array([lag_vals[peak_idx(lag_vals, curve_vals)] for lag_vals, curve_vals in subject_curves])
        se = float(np.std(peak_lags) / np.sqrt(len(peak_lags)))
        mean_pidx = peak_idx(common_lags, mean_curve)
        ax.errorbar(common_lags[mean_pidx], mean_curve[mean_pidx], xerr=se, fmt="none", color="black", capsize=4,
                    linewidth=1.5, zorder=6)

        ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
        ax.set_xlim(common_lags.min(), common_lags.max())
        ax.set_title(joint_label, fontweight="bold", fontfamily="Metropolis")

        if show_legend:
            ax.legend(fontsize=6, ncol=2, loc="upper left")


    def plot_coherence_by_lag_2x2(acoustic, value_type="z_scores", kinematic_measure=kinematic,
                                  peak_search_range=None, out_dir=None):
        """
        2x2 grid (all joints / head / right hand / left hand), one combined
        figure per acoustic measure. See _draw_subject_lag_panel for the
        per-panel drawing conventions.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, (ax, (joint_label, joints)) in enumerate(zip(axes, JOINT_PANELS.items())):
            _draw_subject_lag_panel(ax, acoustic, joints, joint_label, value_type, kinematic_measure,
                                    peak_search_range, show_legend=(i == 0))

        ylabel = "Coherence (z-score)" if value_type == "z_scores" else "Coherence (raw average)"
        for ax in [axes[0], axes[2]]:
            ax.set_ylabel(ylabel, fontfamily="Metropolis")
        for ax in [axes[2], axes[3]]:
            ax.set_xlabel("Lag (s)", fontfamily="Metropolis")

        for ax in [axes[2], axes[3]]:
            ax.text(0.0, -0.15, "Kinematics lead acoustics", transform=ax.transAxes, ha="left", va="top", fontsize=8)
            ax.text(1.0, -0.15, "Acoustics lead kinematics", transform=ax.transAxes, ha="right", va="top", fontsize=8)

        fig.suptitle(f"{kinematic_measure} × {acoustic}", fontweight="bold", fontfamily="Metropolis")
        plt.tight_layout()

        if out_dir is None:
            out_dir = op.join(PATH_PLOTS, "Coherence", "Audio", "Per subject lag curves")
        os.makedirs(out_dir, exist_ok=True)
        path_png = op.join(out_dir, f"lag_2x2_{kinematic_measure}_{acoustic}_{value_type}.png")
        plt.savefig(path_png, dpi=150)
        plt.close(fig)
        print(f"Saved: {path_png}")


    for acoustic in ["envelope", "intensity", "pitch", "f1", "f2"]:
        plot_coherence_by_lag_2x2(acoustic, value_type="z_scores")

    ###########################################################################
    # 10 - Silhouette of average z-score coherence per joint
    ###########################################################################
    from krajjat.plot_functions import plot_silhouette

    # Full body (not just UPPER_BODY_JOINTS_REREF): a silhouette is exactly the
    # kind of figure that makes the "coherence is elevated even at anatomically
    # implausible joints (down to the legs/feet)" point from the limitations
    # discussion visible at a glance. Swap to list(UPPER_BODY_JOINTS_REREF) if
    # you'd rather match the scope of every other analysis in this pipeline
    # instead.
    SILHOUETTE_JOINTS = list(ALL_JOINTS)
    #
    # # Same freq/lag convention as #3 (joint ranking) -- this silhouette is the
    # # visual companion to that table, not a new analysis.
    SILHOUETTE_LAG = [0.0]
    SILHOUETTE_FREQ = [0.6]
    kinematic = "acceleration"
    value_type = "averages"

    def _freq_tag(freq):
        """Turn a frequency spec (float, (lo, hi) tuple, or list of both) into a filename-safe tag.

        Examples: [0.6] -> "0.6Hz", [(0.4, 2)] -> "0.4-2Hz", [0.6, (0.4, 2)] -> "0.6Hz+0.4-2Hz".
        """
        if isinstance(freq, (int, float, tuple)):
            freq = [freq]
        parts = []
        for f in freq:
            if isinstance(f, tuple):
                lo, hi = sorted(float(x) for x in f)
                parts.append(f"{lo:g}-{hi:g}Hz")
            else:
                parts.append(f"{float(f):g}Hz")
        return "+".join(parts)

    def plot_joint_silhouette(acoustic, kinematic_measure=kinematic, value_type="z_scores", out_dir=None):
        """
        Computes average coherence per joint (across subjects) for one acoustic
        measure, and renders it as a silhouette via krajjat's plot_silhouette --
        one circle per joint, colored by that joint's average coherence.
        joint_layout="auto" (the default) picks Kinect vs. Qualisys automatically
        based on whether "Chest" is among the joint names.
        """
        df = build_coherence_tidy_dataframe(subject=subjects, freq=SILHOUETTE_FREQ, joints=SILHOUETTE_JOINTS,
                                            kinematic_measure=[kinematic_measure], acoustic_measure=[acoustic],
                                            lag=SILHOUETTE_LAG, average=kwargs.get("average", "subject"),
                                            value_type=value_type, )
        df = df.copy()
        df["dv"] = df["coherence"]
        df = df.dropna(subset=["dv"])
        joint_means = df.groupby("joint")["dv"].mean()

        plot_dictionary = joint_means.to_dict()

        if out_dir is None:
            out_dir = op.join(PATH_PLOTS, "Coherence", "Audio", "Silhouettes")
        os.makedirs(out_dir, exist_ok=True)
        freq_tag = _freq_tag(SILHOUETTE_FREQ)
        path_png = op.join(out_dir, f"silhouette_{kinematic_measure}_{acoustic}_{value_type}_{freq_tag}_"
                                    f"{kwargs.get('average', 'subject')}.png")

        title_scale = "Coherence (z-score)" if value_type == "z_scores" else "Coherence (raw average)"

        plot_silhouette(plot_dictionary, title=None, title_scale=title_scale, show_values=True, value_format=".3f",
                        font_size=0.5, path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"),
                        show=False, path_save=path_png, verbosity=0, color_scheme="celsius", resolution=(600, 1080))

        print(f"Saved: {path_png}")


    for acoustic in ["envelope", "intensity", "pitch", "f1", "f2"]:
        plot_joint_silhouette(acoustic, kinematic_measure=kinematic, value_type=value_type)

    # ###########################################################################
    # # 11 - Bootstrap CI on the empirical peak lag (no curve-shape assumption)
    # ###########################################################################
    N_BOOT = 2000

    JOINT_PANELS = {"All upper-body joints": list(UPPER_BODY_JOINTS), "Head": ["HeadFront"],
                    "Right hand": ["HandInRight"], "Left hand": ["HandInLeft"]}

    def bootstrap_peak_lag(acoustic, joints, joint_label, kinematic_measure=kinematic, value_type="z_scores",
                           n_boot=N_BOOT, seed=0):
        """
        Resamples subjects with replacement n_boot times; for each resample,
        computes the mean lag-coherence curve and records where its ACTUAL
        maximum lands (same empirical-peak convention as the plots -- no
        parabola fit, no shape assumption). The 2.5th/97.5th percentiles of
        that distribution give a 95% CI on the peak's location.

        p-value: two-tailed bootstrap p, p = 2 * min(P(peak <= 0), P(peak >= 0))
        capped at 1 -- lets this go through the same BH correction as every
        other table in this pipeline, rather than just a binary CI-excludes-0
        flag.
        """

        df = build_coherence_tidy_dataframe(subject=subjects, freq=freq, joints=joints,
                                            kinematic_measure=[kinematic_measure], acoustic_measure=[acoustic],
                                            lag=lag_values, average=kwargs.get("average", "subject"),
                                            value_type=value_type, )
        df = df.copy()
        df["dv"] = df["coherence"]
        df = df.dropna(subset=["dv"])
        df = df.groupby(["subject", "lag"], observed=True)["dv"].mean().reset_index()
        df["lag_display"] = -df["lag"]

        subj_list = sorted(df["subject"].unique())
        n_subj = len(subj_list)

        pivot = df.pivot(index="subject", columns="lag_display", values="dv").loc[subj_list]
        lag_grid = pivot.columns.to_numpy()
        data_matrix = pivot.to_numpy()  # shape (n_subj, n_lags)

        rng = np.random.default_rng(seed)
        boot_peaks = np.empty(n_boot)

        for b in range(n_boot):
            sample_idx = rng.integers(0, n_subj, size=n_subj)
            resampled_mean = data_matrix[sample_idx].mean(axis=0)
            boot_peaks[b] = lag_grid[np.argmax(resampled_mean)]

        ci_low, ci_high = np.percentile(boot_peaks, [2.5, 97.5])
        observed_peak = lag_grid[np.argmax(data_matrix.mean(axis=0))]

        p_below = np.mean(boot_peaks <= 0)
        p_above = np.mean(boot_peaks >= 0)
        p_value = min(1.0, 2 * min(p_below, p_above))

        print(f"Bootstrap peak-lag CI — {kinematic_measure} × {acoustic} — {joint_label}: "
              f"peak = {observed_peak:.3f}, 95% CI = [{ci_low:.3f}, {ci_high:.3f}], p = {p_value:.4f}")

        return observed_peak, ci_low, ci_high, p_value

    for joint_label, joints in JOINT_PANELS.items():
        ps, peaks, cis_low, cis_high, acoustic_names = [], [], [], [], []

        for acoustic in ["envelope", "intensity", "pitch", "f1", "f2"]:
            peak, ci_low, ci_high, p = bootstrap_peak_lag(acoustic, joints, joint_label)
            peaks.append(peak)
            cis_low.append(ci_low)
            cis_high.append(ci_high)
            ps.append(p)
            acoustic_names.append(acoustic)

        ps_corr = _bh_correct(ps)

        if kwargs.get("format_output", "console") == "tabs":
            print(f"Bootstrap peak-lag CI across acoustic measures — {joint_label} "
                  f"(BH-corrected, family = these 5 acoustic measures)")
            print("\t".join(["Acoustic measure", "Peak lag", "95% CI low", "95% CI high", "p", "p (p-corr)"]))
            for name, peak, lo, hi, p, pc in zip(acoustic_names, peaks, cis_low, cis_high, ps, ps_corr):
                print("\t".join([name, f"{peak:.3f}", f"{lo:.3f}", f"{hi:.3f}", _format_p_tabs(p),
                                 _format_p_tabs(pc), ]))
            print()
        else:
            print(f"\n---- Bootstrap peak-lag CI across acoustic measures — {joint_label} "
                  f"(BH-corrected, family = these 5) ----")
            print(f"{'Acoustic measure':<16} | {'Peak lag':>10} | {'95% CI':>18} | {'p':>10} | {'p (p-corr)':>10}")
            for name, peak, lo, hi, p, pc in zip(acoustic_names, peaks, cis_low, cis_high, ps, ps_corr):
                print(f"{name:<16} | {peak:>10.3f} | [{lo:>6.3f}, {hi:>6.3f}] | "
                      f"{_p_digestible_colored(p):>10} | {_p_digestible_colored(pc):>10}")

    plot_coherence_z_scores_frequency_bands(kinematic)
    plot_z_scores_lags(kinematic)