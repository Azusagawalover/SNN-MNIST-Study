import torch
import torch.nn as nn
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen

T = 20
beta = 0.9
batch_size = 1

transform = transforms.ToTensor()
dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
image, label = next(iter(loader))

print("真实标签:", label.item())
print("原始图片 shape:", image.shape)

spike_input = spikegen.rate(image, num_steps=T)
print("Spike input shape:", spike_input.shape)

fc1 = nn.Linear(28 * 28, 128)
lif1 = snn.Leaky(beta=beta)
fc2 = nn.Linear(128, 10)
lif2 = snn.Leaky(beta=beta)

mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()
output_spikes = []

for t in range(T):
    x = spike_input[t].view(batch_size, -1)
    current1 = fc1(x)
    spk1, mem1 = lif1(current1, mem1)
    current2 = fc2(spk1)
    spk2, mem2 = lif2(current2, mem2)
    output_spikes.append(spk2)

output_spikes = torch.stack(output_spikes)
print("Output spike shape:", output_spikes.shape)
spike_count = output_spikes.sum(dim=0)
print("10个输出神经元的 Spike 数量:")
print(spike_count)
prediction = spike_count.argmax(dim=1)
print("网络预测:", prediction.item())
print("说明：本脚本未训练模型，因此预测通常没有意义。")
