import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen, surrogate
from snntorch import functional as SF

beta = 0.9
batch_size = 128
learning_rate = 1e-3
num_steps = 50
epoch_list = [2, 5, 10]
repeat_times = 3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用设备:", device)

transform = transforms.ToTensor()
train_dataset = Subset(datasets.MNIST("./data", train=True, download=True, transform=transform), range(10000))
test_dataset = Subset(datasets.MNIST("./data", train=False, download=True, transform=transform), range(2000))

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
        outputs = []
        for t in range(num_steps):
            x_t = spike_input[t].view(spike_input[t].size(0), -1)
            spk1, mem1 = self.lif1(self.fc1(x_t), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            outputs.append(spk2)
        return torch.stack(outputs)

def run_single_experiment(num_epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = SNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = SF.ce_count_loss()
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        correct = total = 0
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prediction = output.sum(dim=0).argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {correct / total * 100:.2f}%"
        )

    training_time = time.time() - start
    model.eval()
    correct = total = 0
    total_output_spikes = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            prediction = output.sum(dim=0).argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
            total_output_spikes += output.sum().item()

    return correct / total, training_time, total_output_spikes / total

all_results = {}
for num_epochs in epoch_list:
    accs, times, spikes = [], [], []
    for repeat in range(repeat_times):
        acc, t, spk = run_single_experiment(num_epochs, 42 + repeat)
        accs.append(acc)
        times.append(t)
        spikes.append(spk)
    all_results[num_epochs] = {
        "accuracy": np.array(accs),
        "time": np.array(times),
        "spikes": np.array(spikes),
    }

print("T = 50 不同训练轮数最终统计结果")
print("Epochs | Accuracy(mean±std) | TrainTime(mean±std) | AvgSpikes(mean±std)")
summary = []
for num_epochs in epoch_list:
    acc = all_results[num_epochs]["accuracy"]
    tim = all_results[num_epochs]["time"]
    spk = all_results[num_epochs]["spikes"]
    row = (num_epochs, acc.mean(), acc.std(), tim.mean(), tim.std(), spk.mean(), spk.std())
    summary.append(row)
    print(
        f"{num_epochs:6d} | {row[1] * 100:6.2f}% ± {row[2] * 100:5.2f}% | "
        f"{row[3]:6.2f}s ± {row[4]:5.2f}s | {row[5]:6.2f} ± {row[6]:5.2f}"
    )

plt.figure(figsize=(8, 5))
plt.errorbar(
    [x[0] for x in summary],
    [x[1] * 100 for x in summary],
    yerr=[x[2] * 100 for x in summary],
    marker="o",
    capsize=5,
)
plt.xlabel("Training Epochs")
plt.ylabel("Test Accuracy (%)")
plt.title("T=50: Training Epoch Ablation")
plt.xticks(epoch_list)
plt.grid()
plt.tight_layout()
plt.show()
