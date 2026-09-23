"""Shared PR-style plotting defaults for the reconstructed figures."""
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

# Allow remote execution on a login node where the project checkout lives elsewhere.
ROOT = Path(os.environ.get("PAPER_FIGURE_ROOT",
                           str(Path(__file__).resolve().parents[1])))
BLUE = "#1769aa"
TEAL = "#0b7285"
ORANGE = "#d95f02"
RED = "#b2182b"
INK = "#17202a"
CMAP = "RdBu_r"
TARGET_WIDTH_PX = 4802

# Global type scale. All pt sizes below are the manuscript baseline multiplied by FONT_SCALE.
# 2026-09-17: user asked for "all fonts larger" at 1.5x, applied globally.
FONT_SCALE = 1.5

def setup():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5 * FONT_SCALE,
        "axes.labelsize": 9.5 * FONT_SCALE, "axes.titlesize": 8 * FONT_SCALE,
        "xtick.labelsize": 7 * FONT_SCALE, "ytick.labelsize": 7 * FONT_SCALE,
        "legend.fontsize": 6.8 * FONT_SCALE,
        "axes.linewidth": 0.7, "lines.linewidth": 1.2, "figure.dpi": 160,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "savefig.dpi": 320, "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#455a64", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })

def finish(fig, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), facecolor='white')
    fig.savefig(out.with_suffix('.svg'), facecolor='white')
    normalize_png_width(out.with_suffix('.png'))
    plt.close(fig)

def normalize_png_width(path):
    """Mechanically enforce the manuscript raster width while preserving aspect."""
    path = Path(path)
    with Image.open(path) as image:
        if image.width == TARGET_WIDTH_PX:
            return
        height = round(image.height * TARGET_WIDTH_PX / image.width)
        image.resize((TARGET_WIDTH_PX, height), Image.Resampling.LANCZOS).save(path)

def panel_label(ax, label):
    ax.text(-0.12, 1.05, f"({label})", transform=ax.transAxes,
            fontweight='bold', va='bottom', ha='left', fontsize=11.5 * FONT_SCALE)
