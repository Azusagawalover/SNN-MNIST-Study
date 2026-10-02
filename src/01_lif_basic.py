import numpy as np
import matplotlib.pyplot as plt

T = 100
beta = 0.9
threshold = 1.0
input_current = np.ones(T) * 0.2

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

spike_times = np.where(np.array(spike_history) == 1)[0]

plt.figure(figsize=(8, 5))
plt.plot(membrane_history)
for t in spike_times:
    plt.axvline(t, linestyle="--")
plt.xlabel("Time step")
plt.ylabel("Membrane potential")
plt.title("LIF Neuron")
plt.tight_layout()
plt.show()
