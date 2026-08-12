import numpy as np
from PIL import Image

# Per-channel HU windows MedGemma 1.5's CT preprocessing maps to RGB - arXiv:2604.05081.
CT_WINDOWS = {
    "R": (-1024, 1024),
    "G": (-135, 215),
    "B": (0, 80),
}

# MedGemma 1.5 was trained/evaluated with at most 85 axial slices per query
# (21,760 vision tokens at ~256 tokens/slice) - arXiv:2604.05081.
MAX_SLICES = 85


def hu_to_rgb(hu):
    """Map one 2D HU slice (numpy array) to the 3-channel window MedGemma 1.5 expects."""
    channels = []
    for lo, hi in CT_WINDOWS.values():
        clipped = np.clip(hu, lo, hi)
        channels.append(((clipped - lo) / (hi - lo) * 255).astype(np.uint8))
    return Image.fromarray(np.stack(channels, axis=-1), mode="RGB")


def sample_slices(items, max_slices=MAX_SLICES):
    """Equidistant sampling down to MedGemma 1.5's training-time slice cap."""
    if len(items) <= max_slices:
        return items
    indices = np.linspace(0, len(items) - 1, max_slices).round().astype(int)
    return [items[i] for i in indices]
