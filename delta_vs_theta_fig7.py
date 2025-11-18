import matplotlib.pyplot as plt
import numpy as np

# Generate noise levels (SNR in dB)
snr = np.linspace(-10, 10, 400)

# Theta oscillations: linear decrease with noise
theta_strength = np.clip(0.05 * (snr + 10), 0, 1)  # Red linear decline

# Delta oscillations: sigmoid with sharper drop below 0 dB
delta_strength = 1 / (1 + np.exp(-0.8 * (snr + 1)))  # Orange sigmoid

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(snr, delta_strength, label='Delta Oscillations', color='orange', linewidth=2)
plt.plot(snr, theta_strength, label='Theta Oscillations', color='red', linewidth=2)

# Labels and title
# plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Signal-to-Noise Ratio')
plt.ylabel('Oscillation Strength')
plt.legend()
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',
    left=False,
    labelleft=False, # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
# plt.grid(True)
plt.tight_layout()
plt.show()
