# 已记录实验结果

本目录保存本次学习过程中已经实际跑出的两组核心消融实验结果：

- `t50_epoch_ablation.csv`：固定 `T=50`，比较 2 / 5 / 10 个训练 epoch。
- `time_step_fair_ablation.csv`：固定 `Epoch=10`，比较 `T=5 / 10 / 20 / 50`。

注意：`AvgOutputSpikes` 只统计输出层 10 个神经元，不可直接等同于芯片能耗。
