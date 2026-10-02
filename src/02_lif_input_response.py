import numpy as np
import matplotlib.pyplot as plt

T = 100
beta = 0.9
threshold = 1.0

input_current = np.zeros(T)
input_current[0:30] = 0.08
input_current[30:70] = 0.25
input_current[70:100] = 0.12

membrane = 0.0
membrane_history = []
spike_history = []

for t in range(T):
    membrane = beta * membrane + input_current[t]
    spike = 0
    if membrane >= threshold:
        spike = 1
        membrane = 0.0
    membrane_history.append(membrane)
    spike_history.append(spike)

membrane_history = np.array(membrane_history)
spike_history = np.array(spike_history)
spike_times = np.where(spike_history == 1)[0]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(input_current)
axes[0].set_ylabel("Input current")
axes[0].set_title("LIF Neuron Response")

axes[1].plot(membrane_history)
axes[1].axhline(threshold, linestyle="--")
axes[1].set_ylabel("Membrane potential")

axes[2].scatter(spike_times, np.ones_like(spike_times))
axes[2].set_ylabel("Spike")
axes[2].set_xlabel("Time step")
axes[2].set_ylim(0, 1.5)

plt.tight_layout()
plt.show()
