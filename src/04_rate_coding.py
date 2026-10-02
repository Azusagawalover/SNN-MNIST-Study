import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 100
input_values = [0.2, 0.5, 0.8]
spike_trains = []

for value in input_values:
    random_values = np.random.rand(T)
    spikes = (random_values < value).astype(int)
    spike_trains.append(spikes)

for value, spikes in zip(input_values, spike_trains):
    print(
        f"Input = {value}, Spike count = {np.sum(spikes)}, "
        f"Rate = {np.mean(spikes):.2f}"
    )

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for i in range(3):
    spike_times = np.where(spike_trains[i] == 1)[0]
    axes[i].eventplot(spike_times)
    axes[i].set_ylabel(f"x={input_values[i]}")
axes[0].set_title("Rate Coding")
axes[2].set_xlabel("Time step")
plt.tight_layout()
plt.show()
