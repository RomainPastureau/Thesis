from matplotlib import pyplot as plt, gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

from C01_duration_analyses import *


def combined_figure(df_clean):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.5)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, :])

    cartoon_duration_as_predictor(1, df_clean, ax=ax1)
    trial_order(2, df_clean, ax=ax2)
    retelling_duration_by_subject(3, df_clean, ax=ax3)

    fig.canvas.draw()
    pad = 0.02

    fp = FontProperties(family="Merriweather", weight="black", size=16)

    for ax, label, grey in zip([ax1, ax2, ax3], ["A", "B", "C"], [True, False, False]):
        fig.canvas.draw()
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())

        # Add letter first
        letter = fig.text(bbox.x0, bbox.y1 + pad, label,
                          transform=fig.transFigure, fontproperties=fp,
                          va="bottom", ha="left", clip_on=False)

        if grey:
            # Redraw so the letter is included in the new tightbbox
            fig.canvas.draw()
            bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            # Also extend bbox to include the letter manually
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

    save_figure(fig, PATH_FIGURES, "fig_combined.png", **SAVE_KW)
    plt.close(fig)

if __name__ == "__main__":
    df = pd.read_excel(PATH_INFO)
    df_keep = df[df["Keep?"] == "Yes"]
    df_clean = df_keep[["Subject", "Show", "Cartoon", "Session order", "Retelling duration", "Cartoon duration"]].dropna()
    print(f"Total trials after cleaning: {len(df_clean)}\n")
    combined_figure(df_clean)