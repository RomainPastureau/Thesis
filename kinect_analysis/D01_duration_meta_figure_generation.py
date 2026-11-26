from matplotlib import pyplot as plt, gridspec
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

from C01_duration_analyses import *


def combined_figure(df_clean):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.5)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, :3])
    ax4 = fig.add_subplot(gs[1, 3])

    cartoon_duration_as_predictor(1, df_clean, ax=ax1)
    retelling_duration_by_subject(2, df_clean, ax=ax2)
    trial_order(3, df_clean, ax=ax3)
    mask_condition(4, df_clean, ax=ax4)

    fig.canvas.draw()
    pad = 0.02

    fp = FontProperties(family="Merriweather", weight="black", size=16)


    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()

    for ax, label, grey in zip([ax1, ax2, ax3, ax4], ["A", "B", "C", "D"], [True, False, False, True]):
        fig.canvas.draw()
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

    save_figure(fig, PATH_FIGURES, "fig_combined.png", **SAVE_KW)
    plt.close(fig)

if __name__ == "__main__":
    def is_excluded(row):
        return row["Cartoon"] in EXCLUDED_RECORDINGS.get(row["Subject"], [])

    df = pd.read_excel(PATH_INFO, "Videos (excl.)")
    df_clean = df[["Subject", "Show", "Cartoon", "Block", "Session order", "Masked?", "Retelling duration",
                   "Cartoon duration"]].dropna()
    print(f"Total trials before cleaning: {len(df_clean)}\n")
    df_clean = df_clean[~df_clean.apply(is_excluded, axis=1)]
    print(f"Total trials after cleaning: {len(df_clean)}\n")
    combined_figure(df_clean)