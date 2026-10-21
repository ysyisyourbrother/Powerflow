# competition_v1：赛事对齐的 IEEE 39 潮流数据集

本目录是**按比赛输入/输出格式生成**的 IEEE 39 节点数据集，用于预训练和
后续迁移。输入是比赛要求的**节点净注入 P/Q、电压幅值 V 及三通道掩码**；
输出是**全部节点 Vm + 全部支路的 Δθ/Pf/Pt/Qf/Qt**。

它对应现阶段的比赛场景：发电计划、电压设定和网架都固定，**只扰动负荷**。
每个样本对全系统负荷做一个缩放，再叠加逐母线的小波动，然后重新求解潮流。

- schema 版本：`competition.v1`
- 样本数：2000
- 分片：4 × 500
- 拓扑：`case39_base`，`topology_hash = e789dd27ac54d348`
- 生成日期：2026-10-19
- 数据总入口：[`dataset/README.md`](../../../README.md)
- 完整教程（字段、掩码、打标、训练建议）：
  [`competition-dataset-guide.md`](../../../competition-dataset-guide.md)

> 数据文件本身不入库（`.gitignore` 排除了 `samples/`、`topology/`、
> `config.json`、`manifest.json`、`checksums.sha256`、`stats.json`）。
> 只提交本 README 和生成脚本，产物一条命令即可逐位重跑。

## 1. 重新生成

```bash
# 默认：2000 条，mixed 掩码（70% structural + 20% dropout + 10% extra）
python src/examples/build_competition_dataset.py --n-samples 2000

# 只生成标准潮流掩码
python src/examples/build_competition_dataset.py --n-samples 2000 --mask-mode structural

# 只生成 dropout 掩码
python src/examples/build_competition_dataset.py --n-samples 2000 --mask-mode dropout

# 只生成 extra 掩码
python src/examples/build_competition_dataset.py --n-samples 2000 --mask-mode extra
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--n-samples` | 2000 | 样本数 |
| `--master-seed` | 20261019 | 负荷场景随机种子 |
| `--mask-seed` | 7 | 掩码随机种子 |
| `--shard-size` | 500 | 每个分片的样本数 |
| `--out` | `dataset/ieee39/processed/competition_v1` | 输出目录 |
| `--sigma-low` | 0.8 | 全局负荷缩放下界 |
| `--sigma-high` | 1.10 | 全局负荷缩放上界（不得超过负荷裕度 1.1779） |
| `--mask-mode` | mixed | `structural` / `dropout` / `extra` / `mixed` |
| `--dropout-prob` | 0.10 | 每个已知量被隐藏的概率 |
| `--extra-prob` | 0.10 | 每个未知量被揭示的概率 |

生成脚本会依次做：基准复现 → 拓扑/字段检查 → 掩码分布检查 →
标签物理自洽性检查 → 收敛检查 → 写盘 → 读回校验。任一断言失败即中止。

## 2. 目录布局

```text
competition_v1/
├── README.md              # 本文件
├── config.json            # 采样配置、种子、单位约定
├── manifest.json          # 导出时间、分片清单、拓扑摘要
├── stats.json             # 各通道统计量（训练归一化参考）
├── checksums.sha256       # 拓扑包 + 每个分片的 SHA-256
├── topology/
│   └── case39_base.npz    # 拓扑包，所有样本共用
└── samples/
    ├── shard_000000.npz   # 500 条
    ├── shard_000001.npz   # 500 条
    ├── shard_000002.npz   # 500 条
    └── shard_000003.npz   # 500 条
```

校验文件完整性：

```bash
cd dataset/ieee39/processed/competition_v1
sha256sum -c checksums.sha256
```

## 3. 字段与单位

### 3.1 拓扑包 `topology/case39_base.npz`

| 字段 | 形状 | 单位 | 说明 |
|---|---|---|---|
| `node_ids` | [39] int32 | — | 原始母线号 1~39 |
| `edge_index` | [2,46] int32 | — | 0 基节点下标；`[0]=from`，`[1]=to` |
| `branch_ids` | [46] int32 | — | 支路行号 0~45 |
| `branch_features` | [46,3] f8 | p.u. | `[R, X, B/2]`，B/2 是原始 `BR_B / 2`，不再缩放 |
| `branch_type` | [46] int8 | — | 0=线路，1=变压器，2=移相变压器 |
| `node_kv` | [39] f8 | kV | 节点电压等级；case39 全为 345 |
| `from_kv` | [46] f8 | kV | 首端电压等级 |
| `to_kv` | [46] f8 | kV | 末端电压等级 |
| `base_mva` | 标量 | MVA | 100 |
| `nb`、`ne` | 标量 | — | 39、46 |
| `aux_tap` | [46] f8 | p.u. | 有效变比（原始 0 已替换为 1.0） |
| `aux_tap_raw` | [46] f8 | p.u. | 原始变比（0 表示无变压器） |
| `aux_shift_deg` | [46] f8 | 度 | 移相角 |
| `aux_rate_a_mva` | [46] f8 | MVA | 容量 |
| `aux_status` | [46] int8 | — | 投运状态 |
| `aux_bus_type` | [39] int8 | — | 1=PQ，2=PV，3=平衡 |
| `topology_hash` | 标量 str | — | 拓扑哈希 |

### 3.2 样本分片 `samples/shard_*.npz`

首轴为样本。`N=39`、`E=46`。

| 字段 | 形状 | 单位 | 说明 |
|---|---|---|---|
| `node_p_mw` | [N] f8 | MW | 掩码后的有功净注入；未知位置填 0 占位 |
| `node_q_mvar` | [N] f8 | MVar | 掩码后的无功净注入；未知位置填 0 占位 |
| `node_vm_pu` | [N] f8 | p.u. | 掩码后的电压幅值；未知位置填 0 占位 |
| `node_p_mask` | [N] u1 | — | 0=已知，1=待预测 |
| `node_q_mask` | [N] u1 | — | 同上 |
| `node_v_mask` | [N] u1 | — | 同上 |
| `target_vm_pu` | [N] f8 | p.u. | 官方标签：全部节点电压幅值 |
| `target_branch_dtheta_deg` | [E] f8 | 度 | 官方标签：`θ_from − θ_to` |
| `target_pf_mw` | [E] f8 | MW | 官方标签：首端注入支路的有功 |
| `target_pt_mw` | [E] f8 | MW | 官方标签：末端注入支路的有功 |
| `target_qf_mvar` | [E] f8 | MVar | 官方标签：首端注入支路的无功 |
| `target_qt_mvar` | [E] f8 | MVar | 官方标签：末端注入支路的无功 |
| `aux_va_deg` | [N] f8 | 度 | 辅助：节点相角，不参与官方评测 |
| `aux_net_p_mw` | [N] f8 | MW | 辅助：真实有功净注入 |
| `aux_net_q_mvar` | [N] f8 | MVar | 辅助：真实无功净注入 |
| `converged` | [S] bool | — | 是否收敛 |
| `iterations` | [S] int16 | — | 牛顿法迭代次数 |
| `final_norm` | [S] f8 | p.u. | 末次残差无穷范数 |
| `initial_norm` | [S] f8 | p.u. | 初始残差无穷范数 |
| `total_load_mw` | [S] f8 | MW | 总负荷 |
| `loss_mw` | [S] f8 | MW | 网损 |
| `sigma` | [S] f8 | — | 全局负荷缩放系数 |
| `mask_mode` | [S] int8 | — | 0=structural，1=dropout，2=extra |
| `n_known_p` | [S] int16 | — | 已知 P 的节点数 |
| `n_known_q` | [S] int16 | — | 已知 Q 的节点数 |
| `n_known_v` | [S] int16 | — | 已知 V 的节点数 |
| `sample_id` | [S] u8 | — | 样本号 |
| `topology_hash` | [S] str | — | 拓扑哈希 |
| `case_hash` | [S] str | — | 输入哈希，用于去重 |

> **数据泄漏警告**
>
> 项目第一版模型输入只用：`node_p_mw`、`node_q_mvar`、`node_vm_pu`、三个掩码、
> `node_kv`、`edge_index`、`branch_features`、`branch_type`、`from_kv`、`to_kv`。
>
> **不要**把 `aux_net_p_mw`、`aux_net_q_mvar`、`aux_va_deg`（样本分片），
> 以及拓扑包里的 `aux_tap`、`aux_shift_deg`（reader 中对应
> `item["aux"]["tap"]`、`item["aux"]["shift_deg"]`）作为输入。它们分别是
> 真实净注入、真实相角和真实字段表未单列的变压器变比；用作输入会泄漏答案
> 或引入比赛里不存在的信息。`aux_*` 只用于物理复算、误差分析、画图和自检。
> `aux_bus_type` 不属于泄漏：真实字段表明确提供节点类型，只是第一版未用。

## 4. 掩码语义

### 4.1 structural：标准潮流掩码

| 节点类型 | mask_P | mask_Q | mask_V | 角色 |
|---|---:|---:|---:|---|
| PQ（29 个） | 0 | 0 | 1 | P/Q 已知，V 待预测 |
| PV（9 个） | 0 | 1 | 0 | P/V 已知，Q 待预测 |
| 平衡（1 个） | 1 | 1 | 0 | V 已知，P/Q 待预测 |

已知数量固定为 **P 38 / Q 29 / V 10**，对应 67 个未知状态量、
67 个功率方程的方阵问题。

### 4.2 dropout / extra / mixed

- `dropout`：在 structural 基础上，以 `dropout_prob` 隐藏一部分原本已知的量，
  模拟量测缺失，问题可能欠定。
- `extra`：在 structural 基础上，以 `extra_prob` 揭示一部分原本未知的量，
  模拟量测更齐全，问题过定但一致。
- `mixed`：按 0.70 / 0.20 / 0.10 的权重在三者间随机选择。

本轮 2000 条的实际分布：

```text
structural：1384 条（69.2%）
dropout：   419 条（20.95%）
extra：     197 条（9.85%）
```

## 5. 数据统计

以下统计量由 `stats.json` 提供，**基于真实物理量**（不是掩码后的 0 占位）：

| 通道 | 均值 | 标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| 节点 P (MW) | 1.2606 | 358.7120 | -817.8097 | 1532.4701 |
| 节点 Q (MVar) | -1.5401 | 101.2648 | -220.8697 | 648.9201 |
| 节点 Vm (p.u.) | 1.02643 | 0.022379 | 0.955546 | 1.063600 |
| 支路 Δθ (°) | -1.4608 | 3.4907 | -25.5313 | 11.6273 |
| 支路 Pf (MW) | -123.2398 | 346.7112 | -1532.4701 | 740.7264 |
| 支路 Pt (MW) | 124.3086 | 347.2263 | -740.7264 | 1532.4701 |
| 支路 Qf (MVar) | -18.2908 | 67.7214 | -248.7102 | 210.5511 |
| 支路 Qt (MVar) | 16.9851 | 89.1075 | -214.5656 | 648.9201 |

注意：

- `stats.json` 只是归一化参考，落盘数据始终是物理量。提交前必须把模型
  输出反变换回物理单位。
- 统计量来自**真实物理量**（`aux_net_*`、`target_*`），不是带 0 占位的
  `node_p_mw` / `node_q_mvar` / `node_vm_pu`。归一化输入时，建议用掩码
  只对已知位置统计，或明确接受“0 占位会被映射成固定值”，让模型靠掩码忽略它。

## 6. 生成时的自检结果

本轮生成全部通过：

| 检查 | 结果 |
|---|---|
| 基准复现 | Vm ∈ [0.9820, 1.0636]，网损 43.64112576 MW，总负荷 6254.23 MW |
| 与 PYPOWER 对照 | `max|dVm| = 1.11e-15`，`max|dVa| = 1.58e-15` |
| 拓扑字段 | 34 条线路 + 12 条变压器支路；B/2 反推 `BR_B` 一致 |
| 已知输入 vs 真实值 | 最大偏差 `9.99e-09` MW/MVar/p.u. |
| 能量平衡 | `|Σ(Pf+Pt) − 网损|` 最大 `1.96e-11` MW |
| 收敛率 | 2000/2000 |
| 迭代数分布 | 4 次：1356 条；5 次：644 条 |
| 读回校验 | 字段形状、dtype、数值逐项一致 |

样本 0 的元数据：

```text
sigma         = 0.875819
总负荷        = 5587.02 MW
网损          = 51.0865 MW
迭代次数      = 4
mask_mode     = structural
known P/Q/V   = 38 / 29 / 10
```

## 7. 已知局限

1. **单一拓扑。** 2000 条样本共享 `topology_hash = e789dd27ac54d348`，
   不能用于跨拓扑泛化评测，也没有 N-1 开断样本。
2. **只扰动负荷。** 发电计划、电压设定、变压器变比、拓扑都不变。
3. **全部收敛。** 没有病态/不收敛样本；σ 上界 1.10，低于负荷裕度 1.1779。
4. **真实字段表没有单列 tap。** 12 条变压器支路中 11 条变比 ≠ 1。
   字段表给了 L/T、R/X/B2、首末端电压等级和额定容量，但没有变比；
   本数据集把 `tap` 放在 `aux_tap` 里供物理复算，模型如果只用官方特征，
   无法精确复现这些支路的潮流。v1 模型统一按“所有支路变比 = 1”处理，
   不输入 `tap`，先用官方标签完成训练；后续拿到真实变比再微调。
5. **全部 345 kV。** `node_kv`、`from_kv`、`to_kv` 在 case39 上是常数，
   字段虽已按多电压等级设计，但这版数据没有该维度的信息量。
6. **低负荷时出现功率倒送。** 非平衡机发电固定，σ<1 时平衡机可能吸收
   功率，节点相角出现正值（样本 0 最大 21.8°）。这是正常物理现象。

迁移到比赛真实数据时，先逐项确认：节点/支路顺序、发电/负荷/补偿如何合成
净注入、V 的单位与掩码语义、B/2 是否已给好且变压器空值如何处理、
L/T 到内部编码的映射、并联标志、节点类型是否作为输入、Δθ 方向与功率符号、
归一化方式，以及是否另有变压器变比来源。用一条已知支路做符号自检，
再用潮流方程做残差复核。

## 8. 读取示例

```python
from powerflow.dataset import CompetitionDataset

ds = CompetitionDataset("dataset/ieee39/processed/competition_v1")
print(len(ds))  # 2000

item = ds[0]
item["node_features"]            # [39,3]  [P, Q, V]
item["node_masks"]               # [39,3]  [mask_P, mask_Q, mask_V]
item["branch_features"]          # [46,3]  [R, X, B/2]
item["edge_index"]               # [2,46]
item["targets"]["vm_pu"]         # [39]
item["targets"]["dtheta_deg"]    # [46]
item["targets"]["pf_mw"]         # [46]
item["aux"]["va_deg"]            # [39]
item["meta"]["mask_mode"]        # 0/1/2

for batch in ds.iter_batches(batch_size=64, shuffle=True, seed=0):
    print(batch["node_features"].shape)   # [64,39,3]
    print(batch["targets"]["vm_pu"].shape)  # [64,39]
    break
```

数据布局与文档入口：[`dataset/README.md`](../../../README.md)。
