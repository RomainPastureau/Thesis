import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

from C02_kinematic_analyses import *


PATH_COMBINED = op.join(PATH_PLOTS, "Kinematics")
os.makedirs(PATH_COMBINED, exist_ok=True)


def _load_png_into_ax(ax, path):
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis("off")


def _add_panel_labels(fig, axes, labels, grey_panels):
    fp = FontProperties(family="Merriweather", weight="black", size=16)
    pad = 0.02
    fig.canvas.draw()

    for ax, label, grey in zip(axes, labels, grey_panels):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        letter = fig.text(bbox.x0, bbox.y1 + pad, label,
                          transform=fig.transFigure, fontproperties=fp,
                          va="bottom", ha="left", clip_on=False)
        if grey:
            fig.canvas.draw()
            bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            letter_bbox = letter.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            from matplotlib.transforms import Bbox
            full_bbox = Bbox.union([bbox, letter_bbox])
            rect = Rectangle(
                (full_bbox.x0 - 0.5 * pad, full_bbox.y0 - pad),
                full_bbox.width + 1.5 * pad, full_bbox.height + 1.5 * pad,
                transform=fig.transFigure, facecolor="#FDEED4",
                edgecolor="none", zorder=0, clip_on=False
            )
            fig.add_artist(rect)


def combined_figure_1(df_mocap):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.4)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1:])
    ax_C = fig.add_subplot(gs[1, :])

    # Panel A — mean speed by joint (silhouette, pygame → PNG → imshow)
    path_sil_A = os.path.join(PATH_FIGURES, "fig_01_mean_speed_by_joint.png")
    mean_speed_by_joint(1, df_mocap, generate_figure=True, path_save=path_sil_A)
    _load_png_into_ax(ax_A, path_sil_A)

    # Panel B — trial order (all joints)
    trial_order(2, df_mocap, joint=None, ax=ax_B)

    # Panel C — mean speed by subject
    mean_speed_by_subject(3, df_mocap, ax=ax_C)

    _add_panel_labels(fig, [ax_A, ax_B, ax_C],
                      ["A", "B", "C"],
                      [True, False, False])

    save_figure(fig, PATH_COMBINED, "fig_combined_1.png", **SAVE_KW)
    plt.close(fig)


def combined_figure_2(df_mocap, kinematic="velocity"):
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(2, 5, wspace=0.1)

    ax_left_up  = fig.add_subplot(gs[0, 0:3])
    ax_right_up = fig.add_subplot(gs[0, 3:])
    ax_left_down  = fig.add_subplot(gs[1, 0:3])
    ax_right_down = fig.add_subplot(gs[1, 3:])

    # Left panel — PSD body graph
    path_body = os.path.join(PATH_FIGURES, f"fig_98_psd_body_{kinematic}.png")
    path_sil = os.path.join(PATH_FIGURES, f"fig_99_psd_stats_silhouette_{kinematic}.png")
    plot_psd(37, kinematic, "envelope", use_reref=True, use_filtered=True,
             joints=None, xlim=(0, 10), df=df_mocap, force=True,
             joint_layout="qualisys",
             nperseg=256, normalization="subject_peak", band_max_range=(0.2, 10),
             show_average=True, show_stats=True, figsize=(16, 12),
             x_label="Frequency (Hz)", y_label="Normalized\namplitude",
             show=False, path_png=path_body, path_png_silhouette=path_sil,
             include_audio=False, legend=False,
             sns_kwargs={"style": "whitegrid", "rc": {"font.family": "Metropolis"}},
             mpl_kwargs={"hspace": 1, "tick_pad": -4, "xlabel_pad": -1, "ylabel_pad": 0,
                         "title_pad": 3, "title_weight": "bold", "xtick_step": 1, "tight_layout": False})
    _load_png_into_ax(ax_left_up, path_body)
    # Right panel — power distribution silhouette
    _load_png_into_ax(ax_right_up, path_sil)

    # df_mocap = df_mocap[df_mocap["subject"].isin(SUBJECTS_NO_OUTLIERS)]

    path_body_peak = os.path.join(PATH_FIGURES, f"fig_100_psd_body_peak_{kinematic}.png")
    plot_psd(38, kinematic, "envelope", use_reref=True, use_filtered=True, joints=None, xlim=(0, 10), df=df_mocap,
             force=True, joint_layout="qualisys", nperseg=256, normalization="fooof", band_max_range=(0.2, 10),
             subjects=SUBJECTS_NO_OUTLIERS,
             exclude_curves={"WaistFrontLeft": ["S009", "S024"], "AnkleLeft": ["S017"], "HeelLeft": ["S017"], "ForefootOutLeft": ["S017"],
                             "ForefootInLeft": ["S017"], "ToetipLeft": ["S017"], "KneeRight": ["S016"], "ShinRight": ["S016"],
                             "ThighRight": ["S013"], },
             mark_peak_freq=True, peak_freq_range=(0.2, 10), show_average=True, show_stats=True, figsize=(16, 12),
             x_label="Frequency (Hz)", y_label="Normalized\namplitude", show=False, path_png=path_body_peak,
             include_audio=False, legend=False, sns_kwargs={"style": "whitegrid", "rc": {"font.family": "Metropolis"}},
             mpl_kwargs={"hspace": 1, "tick_pad": -4, "xlabel_pad": -1, "ylabel_pad": 0, "title_pad": 3,
                         "title_weight": "bold", "xtick_step": 1, "tight_layout": False})
    _load_png_into_ax(ax_left_down, path_body_peak)

    path_peak_sil = os.path.join(PATH_FIGURES, f"fig_101_peak_freq_silhouette_{kinematic}.png")
    plot_peak_freq_silhouette(39, kinematic, "envelope", use_reref=True, use_filtered=True, joints=None, df=df_mocap,
                              force=True, nperseg=256, normalization="fooof", band_max_range=(0.2, 10), font_size=0.5, color_scheme=["#06B6F9", "#FF8400"],
                              peak_freq_range=(0.2, 10), sd_threshold=1.0, show=False, path_png=path_peak_sil, ignore_sd_threshold=True, resolution=(600, 1080),
                              path_font=op.join(PATH_METROPOLIS, "Metropolis-Regular.otf"))
    _load_png_into_ax(ax_right_down, path_peak_sil)

    _add_panel_labels(fig, [ax_left_up, ax_right_up, ax_left_down, ax_right_down],
                      ["A", "B", "C", "D"],
                      [False, False, False, False])

    save_figure(fig, PATH_COMBINED, f"fig_combined_2_{kinematic}.png", **SAVE_KW)
    plt.close(fig)

def load_dataframe(filtered, kinematic):
    excluded_pairs = {(subject, trial) for subject, trials in EXCLUDED_RECORDINGS.items() for trial in trials}

    if kinematic == "velocity":
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
    elif kinematic == "acceleration":
        if USE_REREF:
            if filtered:
                df = pd.read_parquet(PATH_DATAFRAME_ACCELERATION_REREF_F)
            else:
                df = pd.read_parquet(PATH_DATAFRAME_ACCELERATION_REREF_U)
        else:
            if filtered:
                df = pd.read_parquet(PATH_DATAFRAME_ACCELERATION_F)
            else:
                df = pd.read_parquet(PATH_DATAFRAME_ACCELERATION_U)
    else:
        raise Exception(f"Wrong kinematic type: {kinematic}.")
    print(df.columns.tolist())

    df = df[df["subject"].isin(ALL_SUBJECTS_MINUS_S004)]
    pair_series = pd.Series(list(zip(df["subject"], df["trial"])), index=df.index)
    df = df[~pair_series.isin(excluded_pairs)]
    df_mocap = df[(df["modality"] == "mocap") & (df["measure"] == kinematic)].copy()
    n_unique = df_mocap[["subject", "trial"]].drop_duplicates().shape[0]
    print(f"Unique subject/trial combinations: {n_unique}")

    return df_mocap

if __name__ == "__main__":
    generate_figures = True

    df_mocap = load_dataframe(filtered=False, kinematic="velocity")
    combined_figure_1(df_mocap)
    #
    # kinematic = "acceleration"
    # df_mocap = load_dataframe(filtered=True, kinematic=kinematic)
    # combined_figure_2(df_mocap, kinematic=kinematic)