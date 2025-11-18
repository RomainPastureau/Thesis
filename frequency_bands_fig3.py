import numpy as np
import matplotlib.pyplot as plt

# Parameters for each band: (f_start, f_end, color)
bands = {"delta": (0.5, 3, "orange"),
         "theta": (4, 7, "red"),
         "alpha": (7, 12, "#8802e8"),
         "beta": (12, 30, "#0280e8"),
         "gamma": (30, 100, "#4aa000")}

# Time vector (long enough to see the sweep)
duration = 2.0  # seconds
fs = 1000       # sampling rate
t = np.linspace(0, duration, int(fs * duration))

# Create the subplots
fig, axes = plt.subplots(len(bands), 1, figsize=(10, 8))

for ax, (name, (f_start, f_end, color)) in zip(axes, bands.items()):
    f_t = np.linspace(f_start, f_end, len(t))
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    signal = np.sin(phase)
    
    ax.plot(t, signal, color=color, lw=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

plt.tight_layout()
plt.savefig("waves.png")
plt.show()
