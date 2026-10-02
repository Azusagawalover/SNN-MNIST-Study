# SNN-MNIST-Study

## 从 LIF 神经元到 SNN 分类：基于 MNIST 的脉冲神经网络入门与时间步消融研究

这是一个用于学习 **Spiking Neural Network（SNN，脉冲神经网络）** 的本科入门项目。项目从单个 LIF 神经元开始，逐步实现 Rate Coding、MNIST 脉冲编码、两层 SNN、Surrogate Gradient 训练，以及重复消融实验。

> 项目性质：学习型科研 / 自主实验项目。当前没有提出新的 SNN 方法，重点是理解模型、训练机制与科研实验流程。

## 1. 已完成内容

```text
LIF Neuron
    ↓
Membrane Potential / Spike
    ↓
F-I Curve
    ↓
Rate Coding
    ↓
MNIST → Spike Sequence
    ↓
784 → 128 → 10 SNN
    ↓
Surrogate Gradient
    ↓
MNIST Training
    ↓
Time Step Ablation
    ↓
Repeated Experiments
    ↓
Mean ± Std
```

## 2. 项目结构

```text
SNN-MNIST-Study/
├── README.md
├── requirements.txt
├── CV_project_description.md
├── .vscode/
│   └── settings.json
├── src/
│   ├── 01_lif_basic.py
│   ├── 02_lif_input_response.py
│   ├── 03_fi_curve.py
│   ├── 04_rate_coding.py
│   ├── 05_mnist_rate_coding.py
│   ├── 06_snn_forward.py
│   ├── 07_snn_train.py
│   ├── 08_time_step_ablation.py
│   ├── 09_repeated_ablation.py
│   ├── 10_t50_epoch_ablation.py
│   └── 11_time_step_fair_ablation.py
├── results/
│   ├── t50_epoch_ablation.csv
│   ├── time_step_fair_ablation.csv
│   └── README.md
├── figures/
└── models/
```

## 3. 环境

本项目按以下环境整理：

- macOS / Apple Silicon
- Python 3.9.x
- PyTorch 2.8.x
- torchvision 0.23.x
- snnTorch
- NumPy
- Matplotlib

创建并激活虚拟环境后安装：

```bash
python -m pip install -r requirements.txt
```

当前 VS Code 配置默认解释器：

```text
/Users/mac/snn-env/bin/python
```

如果换电脑，请在 VS Code 中重新选择 Python Interpreter。

## 4. 运行

建议始终从项目根目录运行。例如：

```bash
python src/01_lif_basic.py
python src/05_mnist_rate_coding.py
python src/07_snn_train.py
python src/10_t50_epoch_ablation.py
python src/11_time_step_fair_ablation.py
```

MNIST 会在第一次运行时下载到 `data/`。

## 5. LIF 神经元

离散 LIF 模型：

\[
V_t = \beta V_{t-1} + I_t
\]

当：

\[
V_t \ge V_{th}
\]

神经元产生 Spike 并重置膜电位。

在 `beta=0.9`、`V_th=1` 时，若长期不发放，稳态满足：

\[
V = \frac{I}{1-\beta}=10I
\]

因此当 `I<0.1` 时，稳态膜电位低于阈值，F-I Curve 中基本不会产生 Spike。

## 6. Rate Coding

对输入值：

\[
x\in[0,1]
\]

每个时间步采样：

\[
r\sim U(0,1)
\]

若 `r<x`，则产生 Spike。于是：

\[
Pixel\ Intensity \approx Average\ Firing\ Rate
\]

MNIST 静态图像 `[28,28]` 经编码后变为 `[T,28,28]` 的脉冲序列。

## 7. SNN 网络

```text
MNIST 28×28
    ↓
Rate Coding
    ↓
Flatten 784
    ↓
Linear 784→128
    ↓
128 LIF Neurons
    ↓
Linear 128→10
    ↓
10 LIF Neurons
    ↓
Spike Count
    ↓
argmax → 0~9
```

Spike 阶跃函数无法直接提供有效梯度，因此训练采用 **Surrogate Gradient**。前向传播仍使用真实 0/1 Spike；反向传播使用平滑近似梯度。

## 8. 初步 MNIST 训练

在 10000 张训练图、2000 张测试图、`T=20`、2 个 epoch 的早期实验中得到：

- Epoch 1 Training Accuracy：64.44%
- Epoch 2 Training Accuracy：91.49%
- Test Accuracy：88.85%

该实验主要用于验证 SNN 的训练流程能够正常工作。

## 9. T=50 Epoch Ablation

为判断早期 `T=50` 表现波动是否来自训练不足，固定 `T=50`，分别训练 2 / 5 / 10 个 epoch，并对每组配置使用 3 个随机种子重复实验。

| Epochs | Test Accuracy | Train Time | Avg Output Spikes |
|---:|---:|---:|---:|
| 2 | 85.32% ± 4.57% | 8.70 ± 1.55 s | 10.16 ± 1.41 |
| 5 | 91.02% ± 0.33% | 26.89 ± 5.87 s | 14.31 ± 0.71 |
| 10 | 93.02% ± 0.31% | 54.11 ± 4.88 s | 18.28 ± 1.40 |

结论：增加训练轮数后，准确率明显提高且标准差大幅下降，因此早期 `T=50` 的低准确率与大波动主要与训练不足有关，不能简单归因于较大的时间步本身。

## 10. Fair Time Step Ablation

固定：

- Epoch = 10
- beta = 0.9
- learning rate = 0.001
- 10000 张训练图
- 2000 张测试图
- random seeds = 42, 43, 44

仅改变 `T`：

| Time Steps | Test Accuracy | Train Time | Avg Output Spikes |
|---:|---:|---:|---:|
| 5 | 91.48% ± 0.24% | 12.22 ± 0.60 s | 5.90 ± 0.18 |
| 10 | 93.62% ± 0.19% | 13.92 ± 0.54 s | 13.02 ± 0.36 |
| 20 | **93.83% ± 0.15%** | 19.58 ± 0.08 s | 22.05 ± 2.60 |
| 50 | 93.45% ± 0.19% | 41.59 ± 0.22 s | 18.42 ± 0.99 |

从 `T=5→10`，准确率提高约 2.14 个百分点；从 `T=10→20` 仅提高约 0.21 个百分点；从 `T=20→50`，训练时间超过翻倍，但准确率略有下降。

当前设置下可得到一个初步结论：**约 20 个时间步已经达到接近饱和的分类性能，继续增加时间步会显著增加计算成本，却未带来对应的准确率收益。**

## 11. 结果解释注意事项

- `Avg Output Spikes` 只统计最终输出层 10 个神经元。
- 输出 Spike 数量不能直接等价为真实硬件能耗。
- 若研究效率，应进一步统计隐藏层 Spike、Synaptic Operations、延迟与实际硬件功耗。
- 当前只使用 MNIST 子集和简单两层全连接 SNN，结果不应外推到复杂数据集或不同架构。

## 12. Future Work

- [ ] 统计隐藏层 Spike
- [ ] 估计 Synaptic Operations
- [ ] ANN vs SNN 公平对比
- [ ] beta 消融实验
- [ ] Surrogate Gradient 对比
- [ ] 编码方式对比
- [ ] Spiking CNN
- [ ] CIFAR-10
- [ ] ANN-to-SNN Conversion
- [ ] Spiking Transformer
- [ ] DVS / 事件相机数据集

## 13. 简历可用一句话

> 基于 PyTorch/snnTorch 从零实现 LIF、Rate Coding 与两层 SNN，使用替代梯度完成 MNIST 分类，并通过多随机种子的 Time Step / Epoch 消融实验分析准确率、训练成本与输出脉冲活动之间的关系。
