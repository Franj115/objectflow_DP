# ObjectFlow-Conditioned Diffusion Policy

这是一个面向机器人操作的 ObjectFlow 条件扩散策略。系统先根据近期视觉观测预测物体未来的空间变化，再将预测结果作为条件生成连续动作序列，适用于 Franka Panda、Robotiq 二指夹爪及相同数据接口的机器人平台。

## 设计概览

整体流程分为 ObjectFlow 预测和动作生成两部分：

```text
最近 3 帧场景图像 + 任务指令
        ↓
ObjectFlowPredictor
        ↓
未来 16 帧物体区域 Mask
        ↓
双视角图像 + 机器人状态 + 任务指令 + ObjectFlow
        ↓
Diffusion Policy
        ↓
未来 16 步、每步 7 维的动作序列
```

### ObjectFlowPredictor

ObjectFlowPredictor 使用 DINOv2 提取图像特征、BERT 编码任务指令，并通过 Transformer 和未来时间查询预测连续 16 帧物体 Mask。输出保留每个未来时刻的空间位置，可用于描述目标物体的移动方向和区域变化。

### Flow-Conditioned Diffusion Policy

策略输入包括：

- 场景相机最近 3 帧图像；
- 腕部相机最近 3 帧图像；
- 最近 3 个机器人状态；
- 任务语言指令；
- 未来 16 帧 ObjectFlow Mask。

每张 Flow Mask 先经过共享 CNN 编码器，再加入时间编码并送入时序 Transformer。Flow 特征与双视角视觉特征、语言特征和机器人状态特征融合，作为一维 CNN U-Net 的全局条件。DDPM 从随机噪声逐步生成形状为 `16 × 7` 的动作块：

```text
[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
```

## 主要文件

```text
phase2_core.py                  数据集、特征编码器、Flow Encoder、DDPM
train_flow_conditioned_dp.py    ObjectFlow 条件策略训练
train_dp_baseline.py            普通 Diffusion Policy 训练
infer_policy.py                 动作序列推理
requirements.txt                Python 依赖
```

## 环境安装

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

DINOv2 通过 `torch.hub` 加载，BERT 通过 Transformers 加载。模型文件下载后会自动使用本地缓存。

## 数据格式

机器人数据目录：

```text
DATA_ROOT/
├── language_instruction.json
├── epi_000/
│   ├── scene_camera.mp4
│   ├── wrist_camera.mp4
│   └── trajectory.pkl
├── epi_001/
└── ...
```

`trajectory.pkl` 中使用与视频帧对齐的状态和动作：

```text
pose:   [N, 7]
action: [N, 7]
```

训练集和验证集按完整 Episode 划分。状态和动作的归一化统计量只从训练 Episode 计算，并保存在策略 checkpoint 中。

如果使用真实 Flow Mask，目录格式为：

```text
FLOW_ROOT/
├── episode_000/
│   └── masks/
├── episode_001/
└── ...
```

## 训练 ObjectFlow 条件策略

准备 ObjectFlow checkpoint 后，在当前目录运行：

```bash
python train_flow_conditioned_dp.py \
  --data-root /path/to/robot_dataset \
  --train-episodes 000-013 \
  --val-episodes 014-016 \
  --flow-source predicted \
  --object-flow-project ../object_flow_project \
  --object-flow-checkpoint ../object_flow_project/runs/your_run/best.pt \
  --output-dir runs/flow_conditioned \
  --amp
```

训练期间 ObjectFlowPredictor、DINOv2 和 BERT 保持冻结，策略更新观测投影层、Flow Encoder、条件融合层和 DDPM 动作生成网络。

使用数据集中的 Flow Mask 时：

```bash
python train_flow_conditioned_dp.py \
  --data-root /path/to/robot_dataset \
  --flow-root /path/to/object_flow_masks \
  --flow-source ground_truth \
  --train-episodes 000-013 \
  --val-episodes 014-016 \
  --output-dir runs/flow_conditioned_gt \
  --amp
```

## 训练普通 Diffusion Policy

```bash
python train_dp_baseline.py \
  --data-root /path/to/robot_dataset \
  --train-episodes 000-013 \
  --val-episodes 014-016 \
  --output-dir runs/baseline \
  --amp
```

## 推理

训练完成后，选择一个 Episode 和当前帧位置生成动作块：

```bash
python infer_policy.py \
  --checkpoint runs/flow_conditioned/best.pt \
  --episode-dir /path/to/robot_dataset/epi_017 \
  --frame-index 100 \
  --output-dir outputs/epi017_t100
```

推理结果包括：

```text
action_samples.npy                 16 × 7 动作序列
action_samples.json                可读动作数据
metadata.json                      推理配置与输入信息
predicted_flow_masks/flow_*.png    未来 16 帧 ObjectFlow Mask
```

生成的动作块可交给机器人控制适配器，并按固定执行步数进行滚动闭环规划。
