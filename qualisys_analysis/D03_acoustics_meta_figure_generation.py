import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

from C03_acoustic_analyses import *
from C03_acoustic_analyses import _load_acoustic_psd_dfs, _infer_acoustic_fs

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


def combined_figure_1_new(filtered="unfiltered", average="trial", normalization="subject_peak", mark_peak_freq=True, **kwargs):
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
        fig.add_subplot(gs_bot[1, 1:4]),  # F1
        fig.add_subplot(gs_bot[1, 5:8]),  # F2
    ]

    for ax, measure in zip(axes, measures):
        plot_acoustic_psd(0, measures=[measure], filtered=filtered, normalization=normalization,
                          freq_min=0.2, freq_max=10, mark_peak_freq=mark_peak_freq, peak_freq_range=(1, 10),
                          show=True, ax=ax, exclude_subjects=["S013", "S023"], show_legend=False, **kwargs)

    save_figure(fig, PATH_FIGURES, f"fig_combined_new_{filtered}_{average}_{normalization}.png", **SAVE_KW)
    plt.close(fig)
    print("Saved: fig_combined_new_{filtered}_{average}_{normalization}.png")

def combined_figure_1_old(filtered="unfiltered"):
    """
    Figure 2 — 2×2 grid of normalized PSD plots, no panel letters, titles kept:
      top-left: intensity   top-right: pitch
      bot-left: F1          bot-right: F2
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
        plot_acoustic_psd_old(n=0, acoustic_measure=measure, show_average=False, freq_min=0, filtered=filtered,
                          show_legend=False, freq_max=10, normalization="fooof", fs = FS,
                          generate_figure=False, ax=ax)

    save_figure(fig, PATH_FIGURES, f"fig_combined_old_{filtered}.png", **SAVE_KW)
    plt.close(fig)
    print("Saved: fig_combined.png")

if __name__ == "__main__":
    # combined_figure_1_old("filtered")
    # combined_figure_1_old("unfiltered")
    combined_figure_1_new("filtered", "concat", "fooof", mark_peak_freq=True)
    # combined_figure_1_new("filtered", "trial", "fooof")
    # combined_figure_1_new("unfiltered", "concat", "fooof")
    # combined_figure_1_new("unfiltered", "trial", "fooof")
    combined_figure_1_new("filtered", "concat", "subject_peak", mark_peak_freq=False, ylim=(0, 1.05))
    # combined_figure_1_new("filtered", "trial", "subject_peak", mark_peak_freq=False)
    # combined_figure_1_new("unfiltered", "concat", "subject_peak", mark_peak_freq=False)
    # combined_figure_1_new("unfiltered", "trial", "subject_peak", mark_peak_freq=False)
    # path_psd = os.path.join(PATH_FIGURES, f"fig33_psd_acoustic.png")
    # plot_acoustic_psd(33, filtered="unfiltered", normalization="fooof", freq_min=0.2, freq_max=10, show_average=True,
    #                   show_legend=False, show=False, path_png=path_psd)