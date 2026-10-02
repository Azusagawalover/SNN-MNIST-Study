from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen, surrogate
from snntorch import functional as SF

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 基本参数
torch.manual_seed(42)
num_steps = 20
beta = 0.9
batch_size = 128
epochs = 2
learning_rate = 1e-3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用设备:", device)

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=str(ROOT / "data"), train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=str(ROOT / "data"), train=False, download=True, transform=transform)
train_dataset = Subset(train_dataset, range(10000))
test_dataset = Subset(test_dataset, range(2000))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        spike_input = spikegen.rate(x, num_steps=num_steps)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        output_spikes = []
        for t in range(num_steps):
            current_input = spike_input[t].view(spike_input[t].size(0), -1)
            cur1 = self.fc1(current_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            output_spikes.append(spk2)
        return torch.stack(output_spikes)

model = SNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = SF.ce_count_loss()

for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        spike_output = model(images)
        loss = loss_fn(spike_output, labels)
        loss.backward()
        optimizer.step()

        spike_count = spike_output.sum(dim=0)
        prediction = spike_count.argmax(dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)
        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} Batch {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    print(f"Epoch {epoch + 1} 训练准确率: {correct / total * 100:.2f}%")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        spike_output = model(images)
        prediction = spike_output.sum(dim=0).argmax(dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)

print(f"测试准确率: {correct / total * 100:.2f}%")

model_path = MODEL_DIR / "snn_mnist.pth"
torch.save(model.state_dict(), model_path)
print("模型已保存:", model_path)

# 可视化一个测试样本
image, label = test_dataset[0]
image_batch = image.unsqueeze(0).to(device)
with torch.no_grad():
    spike_output = model(image_batch)[:, 0, :]
spike_count = spike_output.sum(dim=0)
prediction = spike_count.argmax().item()

print("真实标签:", label)
print("网络预测:", prediction)
print("10个神经元 Spike 数量:")
print(spike_count.cpu())

plt.figure(figsize=(4, 4))
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True: {label}   Predicted: {prediction}")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for neuron in range(10):
    spike_times = torch.where(spike_output[:, neuron] > 0)[0].cpu()
    y = torch.ones_like(spike_times) * neuron
    plt.scatter(spike_times, y, marker="|", s=200)
plt.xlabel("Time step")
plt.ylabel("Output neuron")
plt.yticks(range(10))
plt.title("Output Spikes")
plt.grid()
plt.tight_layout()
plt.show()
