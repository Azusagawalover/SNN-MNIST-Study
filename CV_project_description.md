# 简历 / 套磁可用项目描述

## 中文版

**脉冲神经网络（SNN）入门与实验分析**  
基于 PyTorch / snnTorch 从零实现 LIF 神经元、Rate Coding 与两层 SNN，使用替代梯度完成 MNIST 分类；进一步设计 Time Step 与 Training Epoch 消融实验，并使用 3 个随机种子进行重复实验，以 mean ± std 分析准确率、训练成本与输出脉冲活动之间的关系。在固定 10 个训练 epoch 的实验中，`T=20` 获得 `93.83% ± 0.15%` 测试准确率。

## English

**Spiking Neural Network (SNN) Learning and Ablation Study**  
Implemented LIF neurons, rate coding, and a two-layer SNN with PyTorch/snnTorch, trained on MNIST using surrogate gradients. Designed repeated ablation studies on simulation time steps and training epochs with multiple random seeds, and analyzed the trade-off among test accuracy, training cost, and output spike activity. Under 10 training epochs, `T=20` achieved `93.83% ± 0.15%` test accuracy.
