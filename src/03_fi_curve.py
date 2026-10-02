import numpy as np
import matplotlib.pyplot as plt

T = 200
beta = 0.9
threshold = 1.0
input_values = np.linspace(0.0, 0.5, 50)
spike_counts = []

for input_current in input_values:
    membrane = 0.0
    spike_count = 0
    for _ in range(T):
        membrane = beta * membrane + input_current
        if membrane >= threshold:
            spike_count += 1
            membrane = 0.0
    spike_counts.append(spike_count)

spike_counts = np.array(spike_counts)

plt.figure(figsize=(8, 5))
plt.plot(input_values, spike_counts, marker="o")
plt.xlabel("Input Current")
plt.ylabel("Spike Count")
plt.title("LIF Neuron: Input Current vs Spike Count")
plt.grid()
plt.tight_layout()
plt.show()
