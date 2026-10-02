import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

torch.manual_seed(42)

transform = transforms.ToTensor()
dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

image, label = dataset[0]
print("Label:", label)
print("Original shape:", image.shape)
image = image.squeeze(0)
print("After squeeze:", image.shape)

T = 20
spike_sequence = []
for _ in range(T):
    random_values = torch.rand_like(image)
    spikes = (random_values < image).float()
    spike_sequence.append(spikes)

spike_sequence = torch.stack(spike_sequence)
print("Spike sequence shape:", spike_sequence.shape)

average_spikes = spike_sequence.mean(dim=0)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title(f"Original MNIST: {label}")
axes[0, 0].axis("off")

time_steps = [0, 1, 2, 5, 10, 19]
for ax, t in zip(axes.flat[1:7], time_steps):
    ax.imshow(spike_sequence[t], cmap="gray")
    ax.set_title(f"Spike t={t}")
    ax.axis("off")

axes[1, 3].imshow(average_spikes, cmap="gray")
axes[1, 3].set_title("Average Spike Rate")
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()
