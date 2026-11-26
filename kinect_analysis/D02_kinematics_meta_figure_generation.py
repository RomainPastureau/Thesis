import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

from C02_kinematic_analyses import *


PATH_FIGURES = op.join(PATH_PLOTS, "Kinematics")
os.makedirs(PATH_FIGURES, exist_ok=True)


def _load_png_into_ax(ax, path):
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis("off")


def _add_panel_labels(fig, axes, labels, grey_panels):
    from matplotlib.transforms import Bbox

    fp = FontProperties(family="Merriweather", weight="black", size=16)
    pad = 0.02
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()

    for ax, label, grey in zip(axes, labels, grey_panels):
        # Position the letter from the axes + tick/axis labels only, ignoring legends placed above the axes
        bbox_axes = ax.get_tightbbox(renderer, bbox_extra_artists=[]).transformed(to_fig)
        letter = fig.text(bbox_axes.x0, bbox_axes.y1 + pad, label,
                          transform=fig.transFigure, fontproperties=fp,
                          va="bottom", ha="left", clip_on=False)
        if grey:
            fig.canvas.draw()
            # Background covers everything, legend included, plus the letter
            bbox_full = ax.get_tightbbox(renderer).transformed(to_fig)
            letter_bbox = letter.get_window_extent(renderer).transformed(to_fig)
            full_bbox = Bbox.union([bbox_full, letter_bbox])
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
    ax_C = fig.add_subplot(gs[1, 0:2])
    ax_D = fig.add_subplot(gs[1, 2])

    # Panel A — mean speed by joint (silhouette, pygame → PNG → imshow)
    path_sil_A = os.path.join(PATH_FIGURES, "fig97_silhouette_mean_speed.png")
    mean_speed_by_joint(1, df_mocap, generate_figure=True, path_save=path_sil_A)
    _load_png_into_ax(ax_A, path_sil_A)

    # Panel B — mean speed by subject
    mean_speed_by_subject(2, df_mocap, ax=ax_B)

    # Panel C — trial order (all joints)
    trial_order(3, df_mocap, joint=None, ax=ax_C)

    # Panel D — mean speed by condition (all joints)
    mean_speed_by_condition(4, df_mocap, joint=None, ax=ax_D)

    _add_panel_labels(fig, [ax_A, ax_B, ax_C, ax_D],
                      ["A", "B", "C", "D"],
                      [True, False, False, True])

    save_figure(fig, PATH_FIGURES, "fig_combined_1.png", **SAVE_KW)
    plt.close(fig)


def combined_figure_2(df_mocap):
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 5, wspace=0.1)

    ax_left  = fig.add_subplot(gs[0, 0:3])
    ax_right = fig.add_subplot(gs[0, 3:])

    joints_psd = ["Head", "Neck", "ShoulderRight", "ShoulderLeft", "ElbowRight", "ElbowLeft",
                  "WristRight", "WristLeft", "HandRight", "HandLeft",
                  "SpineShoulder", "SpineMid", "SpineBase", "HipRight", "HipLeft"]

    # Left panel — PSD body graph
    path_body = os.path.join(PATH_FIGURES, "fig98_psd_body.png")
    path_sil = os.path.join(PATH_FIGURES, "fig99_psd_stats_silhouette.png")
    plot_psd(37, "velocity", "envelope", use_reref=True, use_filtered=True,
             joints=joints_psd,
             joint_layout=[[None, None, "Head",          None,          None         ],
                            [None, None, "Neck",          None,          None         ],
                            ["ElbowRight", "ShoulderRight", "SpineShoulder", "ShoulderLeft", "ElbowLeft"],
                            ["WristRight", None,           "SpineMid",    None,          "WristLeft"  ],
                            ["HandRight",  "HipRight",     "SpineBase",   "HipLeft",     "HandLeft"   ]],
             nperseg=256, normalization="subject_peak",
             show_average="top", show_stats=True,
             xlim=[0, 4], x_label="Frequency (Hz)", y_label="Normalized amplitude",
             show=False, path_png=path_body,  path_png_silhouette=path_sil,
             legend=False,
             sns_kwargs={"style": "whitegrid", "rc": {"font.family": "Metropolis"}},
             mpl_kwargs={"hspace": 0.5, "tick_pad": 1})
    _load_png_into_ax(ax_left, path_body)
    # Right panel — power distribution silhouette
    _load_png_into_ax(ax_right, path_sil)

    _add_panel_labels(fig, [ax_left, ax_right],
                      ["A", "B"],
                      [False, False])

    save_figure(fig, PATH_FIGURES, "fig_combined_2.png", **SAVE_KW)
    plt.close(fig)


if __name__ == "__main__":
    # df = pd.read_parquet(PATH_DATAFRAME_VELOCITY_REREF_U if USE_REREF else PATH_DATAFRAME_VELOCITY_U)
    # df_mocap = df[(df["modality"] == "mocap") & (df["measure"] == "velocity")].copy()
    # combined_figure_1(df_mocap)

    df = pd.read_parquet(PATH_DATAFRAME_VELOCITY_REREF_F if USE_REREF else PATH_DATAFRAME_VELOCITY_F)
    df_mocap = df[(df["modality"] == "mocap") & (df["measure"] == "velocity")].copy()
    combined_figure_2(df_mocap)