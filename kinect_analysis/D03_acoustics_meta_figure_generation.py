import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

from C03_acoustic_analyses import *

PATH_FIGURES = op.join(PATH_PLOTS, "Acoustics")
os.makedirs(PATH_FIGURES, exist_ok=True)


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
            letter_bbox = letter.get_window_extent(fig.canvas.get_renderer()).transformed(
                fig.transFigure.inverted())
            from matplotlib.transforms import Bbox
            full_bbox = Bbox.union([bbox, letter_bbox])
            rect = Rectangle(
                (full_bbox.x0 - 0.5 * pad, full_bbox.y0 - pad),
                full_bbox.width + 1.5 * pad, full_bbox.height + 1.5 * pad,
                transform=fig.transFigure, facecolor="#FDEED4",
                edgecolor="none", zorder=0, clip_on=False
            )
            fig.add_artist(rect)


def combined_figure_1():
    """
    Figure 1 — two panels:
      A (left)  : pairwise Spearman correlation heatmap between acoustic measures
      B (right) : F2 values by condition (masked vs unmasked)
    """
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, wspace=0.4, width_ratios=[1.3, 1])

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    acoustic_measure_correlations(n=0, generate_figure=True, ax=ax_A)
    acoustic_measure_by_condition(n=0, acoustic_measure="f2", generate_figure=True, ax=ax_B, p_label_override="n.s.",
                                  unit="Hz")

    _add_panel_labels(fig, [ax_A, ax_B], ["A", "B"], [False, True])

    save_figure(fig, PATH_FIGURES, "fig_combined_01.png", **SAVE_KW)
    plt.close(fig)
    print("Saved: fig_combined_01.png")


def combined_figure_2():
    """
    Figure 2 — 3+2 grid of normalized PSD plots:
      top row   : envelope, intensity, pitch
      bottom row : F1, F2 (centred)
    """
    measures = ["envelope", "intensity", "pitch", "f1", "f2"]

    fig = plt.figure(figsize=(16, 9))
    gs_top = gridspec.GridSpec(2, 9, hspace=0.4, wspace=1.5,
                               left=0.06, right=0.97, top=0.93, bottom=0.08)
    gs_bot = gridspec.GridSpec(2, 9, hspace=0.4, wspace=1.5,
                               left=0.06, right=0.97, top=0.93, bottom=0.08)

    axes = [
        fig.add_subplot(gs_top[0, 0:3]),  # envelope
        fig.add_subplot(gs_top[0, 3:6]),  # intensity
        fig.add_subplot(gs_top[0, 6:9]),  # pitch
        fig.add_subplot(gs_bot[1, 1:4]),  # F1  (cols 1-2 of 6 → centred-left)
        fig.add_subplot(gs_bot[1, 5:8]),  # F2  (cols 3-4 of 6 → centred-right)
    ]

    for ax, measure in zip(axes, measures):
        plot_acoustic_psd(n=0, acoustic_measure=measure, show_average=False,
                          show_legend=False, freq_max=4, normalization="subject_peak",
                          generate_figure=False, ax=ax)

    save_figure(fig, PATH_FIGURES, "fig_combined_02.png", **SAVE_KW)
    plt.close(fig)
    print("Saved: fig_combined_02.png")


if __name__ == "__main__":
    # combined_figure_1()
    combined_figure_2()