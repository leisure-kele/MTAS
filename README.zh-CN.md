# 触觉何时有助于时序动作分割？

论文 *When Does Touch Help? Multimodal Temporal Action Segmentation in Human Bimanual Manipulation* 的代码实现。

[English version](README.md)

## 项目概述

本仓库实现对人类双手操作时序动作分割中实测触觉作用的研究。任务以同步多模态观测为输入，预测 26 类原子动作及其时间转换边界；通过固定的 26→8 映射，进一步分析较高层次的操作阶段。

实验协议在匹配的时序主干内比较纯视觉（V）与视觉—触觉（VT）模型。评价在重叠窗口聚合后的完整序列上进行；在固定模型参数和解码设置后，通过受控的视觉特征缺失与视角不可用条件考察触觉贡献。姿态与派生运动学对照用于检验已有运动状态信息时触觉所提供的补充信息。

## 目录说明

| 目录 | 内容 |
| --- | --- |
| `src/atomseg/` | 序列导出、数据准备、模态编码、时序模型、训练目标、解码与评价实现。 |
| `supplementary/aspnet_style/` | 共享—私有 ASPnet-style 适配融合对照。 |
| `configs/paper/` | 固定的 H0 训练约定与论文分析协议。 |
| `ontologies/` | 26 类原子动作定义及固定的 26→8 阶段映射。 |
| `tests/` | 不依赖公开数据的单元测试与模型结构测试。 |
| `docs/` | 数据发布边界、所需产物与复现实验流程。 |

主要时序模型包含适配后的 MS-TCN++、ASRF-style、ASFormer-style 与 DiffAct-style 主干。M2R2-style 和 ASPnet-style 均是在本项目统一数据接口上的适配对照，不构成对原方法的官方复现。

## 安装

```bash
git clone https://github.com/leisure-kele/MTAS.git
cd MTAS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

如需执行视觉特征导出，还需安装 `export` 可选依赖，并配置官方发布的本地 DINOv3 实现与权重：

```bash
pip install -e '.[dev,export]'
```

论文中的视觉编码器保持冻结。

## 实验协议

- 26 类原子动作，统一在 30 Hz 网格上评价。
- 训练窗口长度为 8 秒，步长为 4 秒。
- 使用三个配对随机种子：`20260824`、`20260825`、`20260826`。
- 仅使用验证集进行检查点选择与解码参数校准。
- 对重叠窗口进行聚合后，在完整序列上计算评价指标。
- 视觉可用性实验中，模型权重与解码参数保持固定。

H0 的标量训练设置见 [`configs/paper/protocol_v3_h0.json`](configs/paper/protocol_v3_h0.json)，分析协议见 [`configs/paper/analysis_plan_v1.json`](configs/paper/analysis_plan_v1.json)。

## 基础检查

```bash
bash scripts/run_unit_checks.sh
python -m atomseg.formal500h_wave1_train --help
python -m atomseg.formal500h_wave4_crossarch_h0 --help
python -m atomseg.formal500h_asformer --help
```

## 复现实验流程

1. 获取发布的数据包及其清单文件。
2. 导出或读取时间同步的 episode-HDF5 序列。
3. 构建按采集组划分的训练、验证、测试产物，并仅依据训练集估计归一化统计量。
4. 在固定协议下训练配对的模态配置。
5. 在验证集上确定模型权重与解码参数。
6. 在完整序列上评价，并执行规定的视觉可用性干预。

具体命令与必要文件见 [docs/reproduction.md](docs/reproduction.md)，数据产物与开放范围见 [docs/data_and_release.md](docs/data_and_release.md)。

## 数据说明

论文使用 44,589 条完整操作序列，总时长为 380.39 小时。本仓库不包含原始记录、参与者图像、触觉流、特征缓存、数据清单、模型权重或私有测试产物。计划发布的 100 小时子集将包含多模态记录、原子动作标注及时间边界；其具体内容与使用条件以数据发布版本的说明为准。除非对应的评价清单明确建立等价关系，不应将公开子集上的结果视为对全文语料结果的数值复现。

## 许可证

代码采用 [MIT License](LICENSE) 发布。数据、派生特征与模型权重适用各自的发布与使用条款。
