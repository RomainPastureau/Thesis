# Add near the top of the file, with the other imports:
import re
import numpy as np
from colorama import init, Style
from itertools import combinations, permutations
from statsmodels.stats.multitest import multipletests

COMPARE_TO_COLUMN = {"joints": "joint", "freq": "freq_label", "kinematic_measures": "kinematic_measure",
                     "acoustic_measures": "acoustic_measure", "lags": "lag", "subject": "subject",
                     "condition": "condition", }


def _bh_correct(pvalues, alpha=0.05, method="fdr_bh"):
    """
    Correct a list of p-values for multiple comparisons (Benjamini-Hochberg by
    default, or any method accepted by statsmodels.stats.multitest.multipletests).

    Non-numeric or NaN entries (e.g. the "reference" level, or a p-value that
    could not be computed) are excluded from the correction entirely -- they
    don't count toward the family size, and they come back as NaN so the
    caller can display them as-is (e.g. "reference") instead of a corrected
    number.

    Parameters
    ----------
    pvalues : list
        p-values to correct, in order. May contain NaN or non-numeric values.
    alpha : float, optional
        Family-wise/FDR level passed to multipletests (does not affect the
        returned corrected p-values themselves, only the reject mask, which
        this function does not return).
    method : str, optional
        Correction method, passed to multipletests. Default "fdr_bh"
        (Benjamini-Hochberg).

    Returns
    -------
    list
        Corrected p-values (float) or NaN, same length and order as `pvalues`.
    """
    pvalues = list(pvalues)
    finite_mask = [isinstance(p, (int, float)) and np.isfinite(p) for p in pvalues]
    finite_pvals = [p for p, m in zip(pvalues, finite_mask) if m]

    corrected = [np.nan] * len(pvalues)

    if len(finite_pvals) == 0:
        return corrected

    _, pvals_corrected, _, _ = multipletests(finite_pvals, alpha=alpha, method=method)

    it = iter(pvals_corrected)
    for i, m in enumerate(finite_mask):
        if m:
            corrected[i] = next(it)

    return corrected



# ---------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------
def _parse_color(color):
    """Accept '#RRGGBB' or (r, g, b)."""
    if isinstance(color, str):
        color = color.strip()
        if color.startswith("#"):
            color = color[1:]
        if len(color) != 6:
            raise ValueError(f"Invalid hex color: {color!r}")
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    if isinstance(color, (tuple, list)) and len(color) == 3:
        return tuple(int(c) for c in color)

    raise TypeError("Colors must be '#RRGGBB' strings or (r, g, b) tuples/lists.")


def _p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_p(p):
    if np.isnan(p):
        return "nan"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _p_colored(p):
    """Color p-values by conventional significance threshold."""
    if np.isnan(p):
        return "nan"
    p_str = _format_p(p)

    if p < 0.001:
        return _rgb_text(p_str, (0, 130, 0))
    if p < 0.01:
        return _rgb_text(p_str, (80, 150, 0))
    if p < 0.05:
        return _rgb_text(p_str, (180, 130, 0))

    return _rgb_text(p_str, (150, 150, 150))


def _interpolate_gradient(value, vmin, vmax, gradient):
    """
    Map a numeric value to an RGB color using a multi-stop gradient.
    """
    colors = [_parse_color(c) for c in gradient]

    if len(colors) < 2:
        raise ValueError("The gradient must contain at least two colors.")

    if np.isnan(value):
        return (160, 160, 160)

    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
        t = min(max(t, 0.0), 1.0)

    n_segments = len(colors) - 1
    segment = min(int(t * n_segments), n_segments - 1)
    local_t = (t * n_segments) - segment

    c0 = colors[segment]
    c1 = colors[segment + 1]

    return tuple(round(c0[i] + local_t * (c1[i] - c0[i])) for i in range(3))


def _rgb_text(text, rgb):
    """Return text formatted with ANSI true-color escape code."""
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}{Style.RESET_ALL}"


def _strip_ansi(text):
    """Remove ANSI escape codes so table alignment works with colored strings."""
    return re.sub(r"\x1b\[[0-9;]*m", "", str(text))


def _visible_len(text):
    """Length of a string as it appears in the terminal, ignoring ANSI codes."""
    return len(_strip_ansi(text))


def _pad_ansi(text, width, align="<"):
    """Pad a possibly-colored string to a visible width."""
    text = str(text)
    n_pad = max(0, width - _visible_len(text))

    if align == ">":
        return " " * n_pad + text
    if align == "^":
        left = n_pad // 2
        right = n_pad - left
        return " " * left + text + " " * right
    return text + " " * n_pad


def _format_p_tabs(p):
    """
    Bare p-value text for tab-separated (Excel-paste) output: no color, no
    "p = " prefix, and the leading zero is KEPT (0.125, not .125) to match
    Excel's plain numeric display rather than the console's APA-style text.
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""

    if np.isnan(p):
        return ""
    if p < 0.001:
        return "< .001"

    return f"{p:.3f}"


def _format_value_tabs(value, decimals=3):
    """
    Bare marginal-mean text for tab-separated (Excel-paste) output: no color,
    leading zero kept, blank (not "—") for missing values so the cell reads
    as empty once pasted rather than containing a stray character.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""

    if np.isnan(value):
        return ""

    return f"{value:.{decimals}f}"


def _format_p_digestible(p, prefix="p"):
    """Compact p-value text for digestible tables."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return f"{prefix} = n/a"

    if np.isnan(p):
        return f"{prefix} = n/a"
    if p < 0.001:
        return f"{prefix} < .001"

    p_text = f"{p:.3f}"
    if p_text.startswith("0."):
        p_text = "." + p_text[2:]

    return f"{prefix} = {p_text}"


def _p_digestible_colored(p, prefix="p"):
    """
    Color p-values for digestible output.

    Grey: non-significant or unavailable.
    Green scale: significant, darker for smaller p-values.
    """
    text = _format_p_digestible(p, prefix=prefix)

    try:
        p = float(p)
    except (TypeError, ValueError):
        return _rgb_text(text, (150, 150, 150))

    if np.isnan(p):
        return _rgb_text(text, (150, 150, 150))
    if p < 0.001:
        return _rgb_text(text, (0, 120, 0))
    if p < 0.01:
        return _rgb_text(text, (50, 150, 0))
    if p < 0.05:
        return _rgb_text(text, (110, 170, 0))

    return _rgb_text(text, (150, 150, 150))


def _format_digestible_value(value, vmin, vmax, gradient, decimals=3):
    """Format and color a marginal mean/coherence value."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return _rgb_text("—", (150, 150, 150))

    if np.isnan(value):
        return _rgb_text("—", (150, 150, 150))

    text = f"{value:.{decimals}f}"
    if text.startswith("0."):
        text = "." + text[2:]
    elif text.startswith("-0."):
        text = "-." + text[3:]

    rgb = _interpolate_gradient(value, vmin, vmax, gradient)
    return _rgb_text(text, rgb)


def _digestible_row(label, anova, lmm, lmm_p=None, label_width=64, value_width=16):
    """Format one row of the digestible comparison table."""
    row = (f"{_pad_ansi(label, label_width)} | "
           f"{_pad_ansi(anova, value_width, align='>')} | "
           f"{_pad_ansi(lmm, value_width, align='>')}")

    if lmm_p is not None:
        row += f" | {_pad_ansi(lmm_p, value_width, align='>')}"

    return row


def _human_effect_name(col):
    """Human-readable name for one dataframe/model column."""
    names = {"kinematic_measure": "Kinematic measure", "acoustic_measure": "Acoustic measure",
        "freq_label": "Frequency", "joint": "Joint", "lag": "Lag", "condition": "Condition", "subject": "Subject", }
    return names.get(col, col.replace("_", " ").title())


def _human_effect_title(effect_cols):
    """Human-readable name for a main effect or interaction."""
    return " × ".join(_human_effect_name(col) for col in effect_cols)


def _effect_cols_from_anova_df(anova_df):
    """Infer tested factor columns from an ANOVA dataframe."""
    ignored = {"dv", "subject"}
    return [col for col in anova_df.columns if col not in ignored]


def _anova_term_for_columns(columns, anova_table):
    """
    Find the corresponding statsmodels ANOVA term for a list of dataframe columns.

    Example:
    ["freq_label", "joint"] -> "C(freq_label):C(joint)"
    """
    columns = list(columns)

    # First try the natural order.
    term = ":".join(f"C({col})" for col in columns)
    if term in anova_table.index:
        return term

    # Then try permutations, in case the formula used another order.
    for perm in permutations(columns):
        term = ":".join(f"C({col})" for col in perm)
        if term in anova_table.index:
            return term

    return None


def _get_anova_effect_p(res_anova, effect_cols):
    """Extract the ANOVA p-value for a main effect or interaction."""
    anova_table = res_anova.get("anova_table")
    if anova_table is None:
        return np.nan

    term = _anova_term_for_columns(effect_cols, anova_table)
    if term is None or term not in anova_table.index:
        return np.nan

    return anova_table.loc[term].get("PR(>F)", np.nan)


def _extract_effect_columns_from_lmm_term(term):
    """
    Extract dataframe columns from an LMM/LRT term.

    Handles terms such as:
    - C(kinematic_measure)
    - C(acoustic_measure, Treatment("envelope"))
    - C(kinematic_measure):C(acoustic_measure)
    - lag
    - I(lag**2)
    """
    term = str(term)

    cols = re.findall(r"C\(\s*([A-Za-z_][A-Za-z0-9_]*)", term)
    if cols:
        return tuple(cols)

    cols = []
    for part in term.split(":"):
        part = part.strip()

        match_power = re.match(r"I\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\*", part)
        if match_power:
            cols.append(match_power.group(1))
            continue

        match_name = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", part)
        if match_name:
            cols.append(match_name.group(1))

    return tuple(cols)


def _get_lmm_effect_p(res_lmm, effect_cols):
    """
    Extract the LMM p-value for a main effect or interaction.

    This uses res_lmm["lrt_table"], so the LMM must have been run with
    run_lrt=True. If no matching LRT row exists, returns NaN.
    """
    lrt_table = res_lmm.get("lrt_table")
    if lrt_table is None or len(lrt_table) == 0:
        return np.nan

    effect_cols = tuple(effect_cols)

    # First pass: exact order match.
    for _, row in lrt_table.iterrows():
        cols = _extract_effect_columns_from_lmm_term(row.get("term", ""))
        if cols == effect_cols:
            return row.get("p", np.nan)

    # Second pass: same columns, regardless of order.
    effect_set = set(effect_cols)
    for _, row in lrt_table.iterrows():
        cols = _extract_effect_columns_from_lmm_term(row.get("term", ""))
        if set(cols) == effect_set:
            return row.get("p", np.nan)

    return np.nan


def _get_lmm_min_fixed_effect_p(res_lmm, effect_cols):
    """
    Return the lowest fixed-effect coefficient p-value matching an effect.

    This is a fallback when the LMM LRT p-value is unavailable.

    Example:
    effect_cols = ("kinematic_measure",)

    It will look for terms such as:
    - C(kinematic_measure)[T.acceleration]
    - C(kinematic_measure)[T.jerk]

    but not interaction terms such as:
    - C(kinematic_measure)[T.jerk]:C(acoustic_measure)[T.pitch]
    """
    model = res_lmm.get("model")
    if model is None:
        return np.nan

    if not hasattr(model, "fe_params") or not hasattr(model, "pvalues"):
        return np.nan

    effect_cols = tuple(effect_cols)
    effect_set = set(effect_cols)

    candidate_pvalues = []

    for term in model.fe_params.index:
        if term == "Intercept":
            continue

        cols = _extract_effect_columns_from_lmm_term(term)

        # Exact effect only.
        # This avoids using interaction p-values as main-effect p-values.
        if tuple(cols) != effect_cols and set(cols) != effect_set:
            continue

        try:
            p = float(model.pvalues.loc[term])
        except Exception:
            continue

        if np.isfinite(p):
            candidate_pvalues.append(p)

    if not candidate_pvalues:
        return np.nan

    return min(candidate_pvalues)


def _get_lmm_effect_p_for_digest(res_lmm, effect_cols):
    """
    Get the LMM p-value to display in the digestible table.

    Priority:
    1. Use the LRT p-value if available.
    2. Otherwise, use the minimum fixed-effect coefficient p-value.
    3. Otherwise, return NaN.

    Returns
    -------
    tuple
        (p_value, source)

        source is one of:
        - "lrt"
        - "min_fixed"
        - "none"
    """
    lrt_p = _get_lmm_effect_p(res_lmm, effect_cols)

    try:
        lrt_p_float = float(lrt_p)
    except (TypeError, ValueError):
        lrt_p_float = np.nan

    if np.isfinite(lrt_p_float):
        return lrt_p_float, "lrt"

    min_fixed_p = _get_lmm_min_fixed_effect_p(res_lmm, effect_cols)

    try:
        min_fixed_p_float = float(min_fixed_p)
    except (TypeError, ValueError):
        min_fixed_p_float = np.nan

    if np.isfinite(min_fixed_p_float):
        return min_fixed_p_float, "min_fixed"

    return np.nan, "none"


def _summarize_effect_values(df, effect_cols, value_col="dv"):
    """Return marginal means for an effect."""
    return (df.groupby(list(effect_cols), observed=True)[value_col].mean().reset_index())


def _merge_effect_summaries(res_anova, res_lmm, effect_cols):
    """Merge ANOVA and LMM marginal means for one effect."""
    anova_df = res_anova["anova_dataframe"]
    lmm_df = res_lmm["lmm_dataframe"]

    anova_summary = _summarize_effect_values(anova_df, effect_cols).rename(columns={"dv": "anova_mean"})
    lmm_summary = _summarize_effect_values(lmm_df, effect_cols).rename(columns={"dv": "lmm_mean"})

    summary = anova_summary.merge(lmm_summary, on=list(effect_cols), how="outer")
    summary["sort_value"] = summary[["anova_mean", "lmm_mean"]].mean(axis=1)

    return summary.sort_values("sort_value", ascending=False)


def _as_list(x, name="parameter"):
    """Convert scalar/list-like parameter to list, while keeping strings atomic."""
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    if isinstance(x, tuple):
        return [x]
    if isinstance(x, list):
        return x
    return [x]


def _infer_context_label(joints):
    """Infer a compact label for the analyzed joint set."""
    joints = _as_list(joints, "joints")

    if joints is None:
        return "Selected data"
    if len(joints) == 1:
        return str(joints[0])

    return "All joints"


def _make_fixed_effect_terms(factor_cols):
    """Return all main effects and interactions from factor columns."""
    terms = []

    for i in range(1, len(factor_cols) + 1):
        for comb in combinations(factor_cols, i):
            terms.append(tuple(comb))

    return terms


def _compare_to_factor_cols(compare):
    """Convert compare aliases to dataframe column names."""
    if compare is None:
        return None

    if isinstance(compare, str):
        compare = (compare,)
    elif isinstance(compare, list):
        compare = tuple(compare)
    elif not isinstance(compare, tuple):
        raise TypeError("`compare` must be a string, list, tuple, or None.")

    return tuple(COMPARE_TO_COLUMN.get(c, c) for c in compare)


def _format_lmm_level_p(p):
    """Format p-values for level rows."""
    if isinstance(p, str) and p == "reference":
        return _rgb_text("reference", (150, 150, 150))

    try:
        p = float(p)
    except (TypeError, ValueError):
        return _rgb_text("—", (150, 150, 150))

    if np.isnan(p):
        return _rgb_text("—", (150, 150, 150))

    return _p_digestible_colored(p, prefix="p")


def _find_lmm_coefficient_p_for_level(res_lmm, effect_cols, row, reference_levels=None):
    """
    Find the fixed-effect coefficient p-value corresponding to a displayed level.

    Returns:
    - np.nan if no matching coefficient exists;
    - "reference" if the row is the reference level;
    - numeric p-value otherwise.
    """
    model = res_lmm.get("model")
    if model is None or not hasattr(model, "pvalues"):
        return np.nan

    if reference_levels is None:
        reference_levels = {}

    # If any part of the displayed level is the reference, there may be no
    # standalone coefficient for this exact row.
    for col in effect_cols:
        if col in reference_levels and row[col] == reference_levels[col]:
            return "reference"

    candidate_terms = []

    for term in model.pvalues.index:
        if term == "Intercept":
            continue

        cols = _extract_effect_columns_from_lmm_term(term)

        if tuple(cols) == tuple(effect_cols) or set(cols) == set(effect_cols):
            candidate_terms.append(term)

    if not candidate_terms:
        return np.nan

    # Match each displayed value against the coefficient term.
    matching_terms = []

    for term in candidate_terms:
        term_str = str(term)
        is_match = True

        for col in effect_cols:
            value = str(row[col])

            # Statsmodels terms usually contain [T.level].
            if f"[T.{value}]" not in term_str:
                is_match = False
                break

        if is_match:
            matching_terms.append(term)

    if not matching_terms:
        return np.nan

    # Usually there should be only one.
    p_values = []

    for term in matching_terms:
        try:
            p = float(model.pvalues.loc[term])
        except Exception:
            continue

        if np.isfinite(p):
            p_values.append(p)

    if not p_values:
        return np.nan

    return min(p_values)

def print_digestible_results(res_anova, res_lmm, subject, freq, joints, kinematic_measure, acoustic_measure, lag,
        compare=None, truncate=10, context_label=None, effects=None,
        gradient=("#40E0D0", "#99CC00", "#FFCC00", "#FFA500", "#FF0000"),
        value_decimals=3, show_factors=True, sort_key=None, ):
    """
    Print a compact ANOVA-vs-LMM digest.

    sort_key : callable or None, optional
        If given, called as sort_key(summary_dataframe) for each effect and
        must return a re-sorted dataframe with the same columns/rows -- lets
        a caller reorder rows (e.g. ascending by frequency) without changing
        anything else about the table. None (default) keeps the existing
        sort-by-coherence-value behavior, so this is fully opt-in and never
        changes output for any existing call site.

    LMM p-values are taken from res_lmm["lrt_table"], so the LMM must be run
    with run_lrt=True if you want LMM effect p-values.
    """
    if context_label is None:
        context_label = _infer_context_label(joints)

    if show_factors:
        print("\n==================== FACTORS ====================")
        print(f"ANOVA formula: {res_anova['formula']}")
        print(f"LMM formula: {res_lmm['formula']}")
        print("")

        if compare is not None:
            compare_text = " × ".join(map(str, compare))
            print(f"Compared factors: {compare_text}")

        print("")

        subject_text = ", ".join(map(str, subject))
        freq_text = ", ".join(str(f) for f in freq)
        joints_text = ", ".join(map(str, joints))
        lag_text = ", ".join(map(str, lag))

        print(f"Subjects: {len(subject)}: {subject_text}")
        print(f"Frequencies: {len(freq)}: {freq_text}")
        print(f"Joints: {len(joints)}: {joints_text}")
        print(f"Kinematic measures: {len(kinematic_measure)}: {', '.join(map(str, kinematic_measure))}")
        print(f"Acoustic measures: {len(acoustic_measure)}: {', '.join(map(str, acoustic_measure))}")
        print(f"Lags: {len(lag)}: {lag_text}")

        if "condition" in res_anova["dataframe"].columns and res_anova["dataframe"]["condition"].notna().any():
            conditions_found = sorted(res_anova["dataframe"]["condition"].dropna().unique())
            print(f"Conditions: {len(conditions_found)}: {', '.join(map(str, conditions_found))}")

    print("\n==================== DIGESTIBLE RESULTS ====================")

    if res_lmm.get("lrt_table") is None:
        print(_rgb_text("Note: LMM p-values are unavailable because res_lmm['lrt_table'] is None. "
                        "Run coherence_lmm(..., run_lrt=True) if you want LMM effect p-values.", (180, 80, 0), ))
        print()

    if effects is None:
        if compare is not None:
            factor_cols = _compare_to_factor_cols(compare)
        else:
            factor_cols = _effect_cols_from_anova_df(res_anova["anova_dataframe"])

        effects = _make_fixed_effect_terms(factor_cols)

    else:
        corrected_effects = []

        for effect in effects:
            if isinstance(effect, str):
                effect = (effect,)
            else:
                effect = tuple(effect)

            corrected_effects.append(tuple(COMPARE_TO_COLUMN.get(col, col) for col in effect))

        effects = corrected_effects

    label_width = 64
    value_width = 16

    print(_digestible_row("Effects", "ANOVA", "LMM", "LMM coef. p", label_width=label_width, value_width=value_width, ))
    print("-" * (label_width + 3 + value_width + 3 + value_width + 3 + value_width))

    for effect_cols in effects:
        effect_cols = tuple(effect_cols)
        title = _human_effect_title(effect_cols)
        title_line = f"{title} " + "-" * max(0, label_width - len(title) - 1)
        print(title_line)

        anova_p = _get_anova_effect_p(res_anova, effect_cols)
        lmm_p, lmm_p_source = _get_lmm_effect_p_for_digest(res_lmm, effect_cols)

        if lmm_p_source == "min_fixed":
            lmm_prefix = "min p"
        else:
            lmm_prefix = "p"

        print(_digestible_row(f"  {context_label}", _p_digestible_colored(anova_p), _p_digestible_colored(lmm_p), "",
            label_width=label_width, value_width=value_width, ))

        summary = _merge_effect_summaries(res_anova, res_lmm, effect_cols)
        if sort_key is not None:
            summary = sort_key(summary)

        all_means = summary[["anova_mean", "lmm_mean"]].to_numpy().ravel()
        all_means = all_means[np.isfinite(all_means)]

        if len(all_means) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.min(all_means))
            vmax = float(np.max(all_means))

        for row_idx, (_, row) in enumerate(summary.iterrows()):
            if truncate is not None and row_idx >= truncate:
                print(f"    ... and {len(summary) - truncate} more.")
                break

            level_label = " × ".join(str(row[col]) for col in effect_cols)

            anova_value = _format_digestible_value(row.get("anova_mean", np.nan), vmin=vmin, vmax=vmax,
                gradient=gradient, decimals=value_decimals, )
            lmm_value = _format_digestible_value(row.get("lmm_mean", np.nan), vmin=vmin, vmax=vmax, gradient=gradient,
                decimals=value_decimals, )

            lmm_level_p = _find_lmm_coefficient_p_for_level(res_lmm=res_lmm, effect_cols=effect_cols, row=row,
                reference_levels=None, )

            lmm_level_p_text = _format_lmm_level_p(lmm_level_p)

            print(
                _digestible_row(f"    {level_label}", anova_value, lmm_value, lmm_level_p_text, label_width=label_width,
                    value_width=value_width, ))

        print()


def print_digestible_results_corrected(res_anova, res_lmm, subject, freq, joints, kinematic_measure, acoustic_measure,
        lag, compare=None, truncate=10, context_label=None, effects=None,
        gradient=("#40E0D0", "#99CC00", "#FFCC00", "#FFA500", "#FF0000"),
        value_decimals=3, show_factors=True, alpha=0.05, method="fdr_bh", format_output="console", sort_key=None, ):
    """
    Same table as print_digestible_results(), with three added columns:
    "ANOVA (p-corr)", "LMM (p-corr)", and "LMM coef. p (p-corr)".

    Correction families (all done with statsmodels.stats.multitest, default
    Benjamini-Hochberg / "fdr_bh"), scoped to THIS table/call only -- nothing
    is pooled across separate calls to this function:

    - "ANOVA (p-corr)": every omnibus ANOVA p-value tested in this table
      (i.e. one per main effect / interaction in `effects`) is corrected
      together, as one family.
    - "LMM (p-corr)": the equivalent omnibus LMM p-values, corrected together
      as their own family (separate from the ANOVA family above).
    - "LMM coef. p (p-corr)": the per-level LMM coefficient p-values (one per
      joint / lag / acoustic measure / etc. shown as a row) are corrected
      WITHIN each effect block separately -- e.g. the 5 acoustic-measure
      level p-values are one family, the 3 kinematic-measure level
      p-values are a different family, and interaction-level p-values (if
      any) are a third. "reference" levels and unavailable (NaN) p-values
      are excluded from correction and simply reprinted as-is.

    This function does not rerun any statistics -- pass in the res_anova /
    res_lmm already returned by coherence_anova() / coherence_lmm(), or by
    run_anova_lmm_digestible(), which returns both.

    format_output : {"console", "tabs"}, optional
        "console" (default): the usual colored terminal table.
        "tabs": a plain tab-separated block, ready to select and paste
        directly into Excel. Column layout mirrors the console table
        (label, Uncorrected ANOVA/LMM/LMM coef., p-corrected ANOVA/LMM/LMM
        coef.), with two differences to match a typical hand-built results
        sheet: (1) numbers keep their leading zero (0.125, not .125) since
        that's how Excel displays plain numeric text; (2) for individual
        LEVEL rows (not the omnibus row), the "LMM" mean column is left
        blank -- it's numerically identical to the "ANOVA" mean column for
        every level row, so only one copy is written, matching the omnibus
        row (whose ANOVA/LMM values are genuinely different tests and are
        both kept). `truncate` is ignored in this mode -- every level is
        always written, since a truncated paste would leave gaps in the
        sheet. `show_factors` and colors are not applicable and are skipped.

    sort_key : callable or None, optional
        If given, called as sort_key(summary_dataframe) for each effect and
        must return a re-sorted dataframe with the same columns/rows -- lets
        a caller reorder the level rows (e.g. ascending by frequency)
        without changing anything else about the table (the "Diff." row,
        the Uncorrected/p-corrected columns, truncate, etc. all stay
        exactly as they'd otherwise be). None (default) keeps the existing
        sort-by-coherence-value behavior, so this is fully opt-in and never
        changes output for any existing call site.
    """
    if format_output not in ("console", "tabs"):
        raise ValueError(f"format_output must be 'console' or 'tabs', got {format_output!r}.")

    if context_label is None:
        context_label = _infer_context_label(joints)

    if format_output == "tabs":
        truncate = None
        show_factors = False
        print(context_label)

    if show_factors:
        print("\n==================== FACTORS ====================")
        print(f"ANOVA formula: {res_anova['formula']}")
        print(f"LMM formula: {res_lmm['formula']}")
        print("")

        if compare is not None:
            compare_text = " × ".join(map(str, compare))
            print(f"Compared factors: {compare_text}")

        print("")

        subject_text = ", ".join(map(str, subject))
        freq_text = ", ".join(str(f) for f in freq)
        joints_text = ", ".join(map(str, joints))
        lag_text = ", ".join(map(str, lag))

        print(f"Subjects: {len(subject)}: {subject_text}")
        print(f"Frequencies: {len(freq)}: {freq_text}")
        print(f"Joints: {len(joints)}: {joints_text}")
        print(f"Kinematic measures: {len(kinematic_measure)}: {', '.join(map(str, kinematic_measure))}")
        print(f"Acoustic measures: {len(acoustic_measure)}: {', '.join(map(str, acoustic_measure))}")
        print(f"Lags: {len(lag)}: {lag_text}")

        if "condition" in res_anova["dataframe"].columns and res_anova["dataframe"]["condition"].notna().any():
            conditions_found = sorted(res_anova["dataframe"]["condition"].dropna().unique())
            print(f"Conditions: {len(conditions_found)}: {', '.join(map(str, conditions_found))}")

    if format_output == "console":
        print(f"\n==================== DIGESTIBLE RESULTS -- BH-corrected within this table (alpha={alpha}) "
              f"====================")

        if res_lmm.get("lrt_table") is None:
            print(_rgb_text("Note: LMM p-values are unavailable because res_lmm['lrt_table'] is None. "
                            "Run coherence_lmm(..., run_lrt=True) if you want LMM effect p-values.", (180, 80, 0), ))
            print()

    if effects is None:
        if compare is not None:
            factor_cols = _compare_to_factor_cols(compare)
        else:
            factor_cols = _effect_cols_from_anova_df(res_anova["anova_dataframe"])

        effects = _make_fixed_effect_terms(factor_cols)
    else:
        corrected_effects = []

        for effect in effects:
            if isinstance(effect, str):
                effect = (effect,)
            else:
                effect = tuple(effect)

            corrected_effects.append(tuple(COMPARE_TO_COLUMN.get(col, col) for col in effect))

        effects = corrected_effects

    label_width = 64
    value_width = 16
    p_corr_width = 20

    # ---- Pass 1: gather every omnibus ANOVA/LMM p-value tested in this
    # table, and correct each set together as its own family. ----
    anova_omnibus_ps = []
    lmm_omnibus_ps = []

    for effect_cols in effects:
        anova_p = _get_anova_effect_p(res_anova, effect_cols)
        lmm_p, _ = _get_lmm_effect_p_for_digest(res_lmm, effect_cols)
        anova_omnibus_ps.append(anova_p)
        lmm_omnibus_ps.append(lmm_p)

    anova_omnibus_ps_corr = _bh_correct(anova_omnibus_ps, alpha=alpha, method=method)
    lmm_omnibus_ps_corr = _bh_correct(lmm_omnibus_ps, alpha=alpha, method=method)

    if format_output == "tabs":
        print("\t".join(["", "Uncorrected", "", "", "p-corrected", "", ""]))
        print("\t".join(["", "ANOVA", "LMM", "LMM coef.", "ANOVA", "LMM", "LMM coef."]))
    else:
        header = (f"{_pad_ansi('Effects', label_width)} | "
                  f"{_pad_ansi('ANOVA', value_width, align='>')} | "
                  f"{_pad_ansi('LMM', value_width, align='>')} | "
                  f"{_pad_ansi('LMM coef. p', value_width, align='>')} | "
                  f"{_pad_ansi('ANOVA (p-corr)', p_corr_width, align='>')} | "
                  f"{_pad_ansi('LMM (p-corr)', p_corr_width, align='>')} | "
                  f"{_pad_ansi('LMM coef. p (p-corr)', p_corr_width, align='>')}")
        print(header)
        print("-" * _visible_len(header))

    for i, effect_cols in enumerate(effects):
        title = _human_effect_title(effect_cols)

        if format_output == "tabs":
            print(title)
        else:
            title_line = f"{title} " + "-" * max(0, label_width - len(title) - 1)
            print(title_line)

        anova_p = anova_omnibus_ps[i]
        lmm_p = lmm_omnibus_ps[i]
        anova_p_corr = anova_omnibus_ps_corr[i]
        lmm_p_corr = lmm_omnibus_ps_corr[i]

        if format_output == "tabs":
            print("\t".join(["Diff.", _format_p_tabs(anova_p), _format_p_tabs(lmm_p), "",
                             _format_p_tabs(anova_p_corr), _format_p_tabs(lmm_p_corr), ""]))
        else:
            omnibus_row = (f"{_pad_ansi(f'  {context_label}', label_width)} | "
                           f"{_pad_ansi(_p_digestible_colored(anova_p), value_width, align='>')} | "
                           f"{_pad_ansi(_p_digestible_colored(lmm_p), value_width, align='>')} | "
                           f"{_pad_ansi('', value_width, align='>')} | "
                           f"{_pad_ansi(_p_digestible_colored(anova_p_corr), p_corr_width, align='>')} | "
                           f"{_pad_ansi(_p_digestible_colored(lmm_p_corr), p_corr_width, align='>')} | "
                           f"{_pad_ansi('', p_corr_width, align='>')}")
            print(omnibus_row)

        summary = _merge_effect_summaries(res_anova, res_lmm, effect_cols)
        if sort_key is not None:
            summary = sort_key(summary)

        all_means = summary[["anova_mean", "lmm_mean"]].to_numpy().ravel()
        all_means = all_means[np.isfinite(all_means)]

        if len(all_means) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.min(all_means))
            vmax = float(np.max(all_means))

        # ---- Pass 2: gather THIS effect's per-level LMM coefficient
        # p-values and correct them together, as their own family (separate
        # from every other effect in this table, and from every other
        # table). "reference" and unavailable p-values are left as-is. ----
        level_rows = list(summary.iterrows())
        raw_level_ps = []

        for _, row in level_rows:
            p = _find_lmm_coefficient_p_for_level(res_lmm=res_lmm, effect_cols=effect_cols, row=row,
                reference_levels=None, )
            raw_level_ps.append(p)

        numeric_level_ps = [p if isinstance(p, (int, float)) else np.nan for p in raw_level_ps]
        level_ps_corr = _bh_correct(numeric_level_ps, alpha=alpha, method=method)

        for row_idx, ((_, row), raw_p, p_corr) in enumerate(zip(level_rows, raw_level_ps, level_ps_corr)):
            if truncate is not None and row_idx >= truncate:
                print(f"    ... and {len(level_rows) - truncate} more.")
                break

            level_label = " × ".join(str(row[col]) for col in effect_cols)
            is_reference = isinstance(raw_p, str) and raw_p == "reference"
            # A level with no usable coefficient p-value (the true reference level, or
            # any lookup that failed to resolve to a number) gets "-" here rather than a
            # blank cell -- blank in a pasted Excel sheet is ambiguous with "no data",
            # while "-" clearly signals "intentionally not applicable".
            is_dash = is_reference or not isinstance(raw_p, (int, float)) or not np.isfinite(raw_p)

            if format_output == "tabs":
                anova_value = _format_value_tabs(row.get("anova_mean", np.nan), decimals=value_decimals)
                coef_p_text = "-" if is_dash else _format_p_tabs(raw_p)
                coef_p_corr_text = "-" if is_dash else _format_p_tabs(p_corr)
                # LMM mean is numerically identical to ANOVA mean for level rows -- left
                # blank here (only the omnibus row keeps both, since those genuinely
                # differ). Corrected ANOVA/LMM mean columns don't apply to level rows.
                print("\t".join([level_label, anova_value, "", coef_p_text, "", "", coef_p_corr_text]))
                continue

            anova_value = _format_digestible_value(row.get("anova_mean", np.nan), vmin=vmin, vmax=vmax,
                gradient=gradient, decimals=value_decimals, )
            lmm_value = _format_digestible_value(row.get("lmm_mean", np.nan), vmin=vmin, vmax=vmax, gradient=gradient,
                decimals=value_decimals, )

            lmm_level_p_text = _format_lmm_level_p(raw_p)

            if is_reference:
                lmm_level_p_corr_text = _rgb_text("reference", (150, 150, 150))
            else:
                lmm_level_p_corr_text = _format_lmm_level_p(p_corr)

            data_row = (f"{_pad_ansi(f'    {level_label}', label_width)} | "
                        f"{_pad_ansi(anova_value, value_width, align='>')} | "
                        f"{_pad_ansi(lmm_value, value_width, align='>')} | "
                        f"{_pad_ansi(lmm_level_p_text, value_width, align='>')} | "
                        f"{_pad_ansi('', p_corr_width, align='>')} | "
                        f"{_pad_ansi('', p_corr_width, align='>')} | "
                        f"{_pad_ansi(lmm_level_p_corr_text, p_corr_width, align='>')}")
            print(data_row)

        print()