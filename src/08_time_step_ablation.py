import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen, surrogate
from snntorch import functional as SF

beta = 0.9
batch_size = 128
epochs = 2
learning_rate = 1e-3
time_steps_list = [5, 10, 20, 50]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("设备:", device)

transform = transforms.ToTensor()
train_dataset = Subset(datasets.MNIST("./data", train=True, download=True, transform=transform), range(10000))
test_dataset = Subset(datasets.MNIST("./data", train=False, download=True, transform=transform), range(2000))
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

    def forward(self, x, num_steps):
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

def run_experiment(num_steps):
    model = SNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = SF.ce_count_loss()
    start = time.time()

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images, num_steps)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

    training_time = time.time() - start
    model.eval()
    correct = total = 0
    total_output_spikes = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images, num_steps)
            prediction = output.sum(dim=0).argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
            total_output_spikes += output.sum().item()

    return correct / total, training_time, total_output_spikes / total

results = []
for T in time_steps_list:
    accuracy, training_time, avg_spikes = run_experiment(T)
    results.append((T, accuracy, training_time, avg_spikes))

print("TimeSteps | Accuracy | TrainTime | AvgSpikes")
for T, accuracy, training_time, avg_spikes in results:
    print(f"{T:9d} | {accuracy * 100:7.2f}% | {training_time:9.2f}s | {avg_spikes:9.2f}")
