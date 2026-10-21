"""赛事对齐的数据集格式：节点 P/Q/V + 掩码，支路 R/X/B2 + 类型，输出全网 Vm 与支路潮流。

本模块负责当前比赛数据集的字段定义、掩码、打标、落盘和读取；负荷场景
采样在 ``sampler`` 模块里。赛题原文里没有写死、但实现时必须选一个约定的
地方，本模块统一如下（完整教程见 ``dataset/competition-dataset-guide.md``，
字段速查见 ``dataset/ieee39/processed/competition_v1/README.md``）：

* 节点 P/Q 是**净注入**（发电减负荷），不是负荷本身；
* P/Q 单位 MW / MVar，V 单位 p.u.，电压等级 kV，R/X/B2 标幺值；
* 支路相角差 ``theta_from - theta_to``，单位度；
* 支路功率是**从该端注入支路**的功率，``pf + pt = 支路有功损耗``；
* 掩码 ``0 = 已知``，``1 = 需要预测``；被掩码的位置填 0.0 作为占位。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from pypower.api import int2ext, ppoption, runpf
from pypower.idx_brch import (
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    RATE_A,
    SHIFT,
    TAP,
    T_BUS,
)
from pypower.idx_bus import BASE_KV, BUS_I, BUS_TYPE, PQ, PV, REF, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG

from ..cases import pad_branch_columns
from ..newton_raphson import solve_newton_raphson
from .sampler import base_load_pu

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查
    from ..cases import PowerFlowProblem

COMPETITION_SCHEMA_VERSION = "competition.v1"

# ---------------------------------------------------------------------------
# 字段顺序：这些元组是"唯一真相来源"，读取器和文档都按它解释列号。
# ---------------------------------------------------------------------------

# 赛题原文：节点特征按 P/Q/V 排列。
NODE_FEATURE_NAMES = ("P", "Q", "V")
NODE_MASK_NAMES = ("mask_P", "mask_Q", "mask_V")

# 赛题原文：支路特征按 R/X/B/2 排列；B/2 使用输入原值。
BRANCH_FEATURE_NAMES = ("R", "X", "B_over_2")

# 赛题要求每条支路输出首末端相角差与首末端 P/Q。
BRANCH_TARGET_NAMES = ("dtheta_deg", "pf_mw", "pt_mw", "qf_mvar", "qt_mvar")

# 掩码取值。
MASK_KNOWN = 0
MASK_PREDICT = 1

# 支路类型。0/1 是最低要求；2 预留给移相变压器，case39 没有。
BRANCH_TYPE_LINE = 0
BRANCH_TYPE_TRANSFORMER = 1
BRANCH_TYPE_PHASE_SHIFTER = 2
BRANCH_TYPE_NAMES = {
    BRANCH_TYPE_LINE: "line",
    BRANCH_TYPE_TRANSFORMER: "transformer",
    BRANCH_TYPE_PHASE_SHIFTER: "phase_shifter",
}

# 掩码策略。structural 是"标准潮流已知量"；另外两种用于鲁棒性增强。
MASK_MODE_STRUCTURAL = 0
MASK_MODE_DROPOUT = 1
MASK_MODE_EXTRA = 2
MASK_MODE_NAMES = {
    MASK_MODE_STRUCTURAL: "structural",
    MASK_MODE_DROPOUT: "dropout",
    MASK_MODE_EXTRA: "extra",
}

# 每个样本的数组字段。
SAMPLE_ARRAY_FIELDS = (
    "node_p_mw",
    "node_q_mvar",
    "node_vm_pu",
    "node_p_mask",
    "node_q_mask",
    "node_v_mask",
    "target_vm_pu",
    "target_branch_dtheta_deg",
    "target_pf_mw",
    "target_pt_mw",
    "target_qf_mvar",
    "target_qt_mvar",
    "aux_va_deg",
    "aux_net_p_mw",
    "aux_net_q_mvar",
)

# 每个样本的标量字段。
SAMPLE_SCALAR_FIELDS = (
    "converged",
    "iterations",
    "final_norm",
    "initial_norm",
    "total_load_mw",
    "loss_mw",
    "sigma",
    "mask_mode",
    "n_known_p",
    "n_known_q",
    "n_known_v",
    "sample_id",
    "topology_hash",
    "case_hash",
)

TOPOLOGY_DIRNAME = "topology"
SAMPLES_DIRNAME = "samples"
CONFIG_FILENAME = "config.json"
MANIFEST_FILENAME = "manifest.json"
CHECKSUMS_FILENAME = "checksums.sha256"

DEFAULT_TOL = 1e-10
DEFAULT_MAX_ITER = 40


def _hash_array(hasher: Any, array: np.ndarray) -> None:
    """把数组按 C 序喂进哈希器，dtype 一并计入避免歧义。"""

    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(contiguous.tobytes())


def topology_hash(problem: "PowerFlowProblem") -> str:
    """由拓扑与支路参数算出的稳定哈希，用作数据集切分的键。

    只取与"连接关系 + 电气参数"有关的列，不取任何潮流解出来的量。
    同一张网架的所有负荷场景共享同一个哈希，可以整体划入训练、验证或
    测试之一，避免同拓扑泄漏。
    """

    hasher = hashlib.sha256()
    bus = problem.bus
    gen = problem.gen
    branch = problem.branch

    # 母线编号 + 类型决定"谁在哪、是什么角色"。
    _hash_array(hasher, bus[:, BUS_I].astype(np.int64))
    _hash_array(hasher, bus[:, BUS_TYPE].astype(np.int64))
    # 发电机接在哪条母线、是否投运。
    _hash_array(hasher, gen[:, GEN_BUS].astype(np.int64))
    _hash_array(hasher, gen[:, GEN_STATUS].astype(np.int64))
    # 支路的连接关系与电气参数。
    for column in (F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, RATE_A, BR_STATUS):
        _hash_array(hasher, branch[:, column].astype(np.float64))
    return hasher.hexdigest()[:16]


def generator_bus_arrays(
    problem: "PowerFlowProblem",
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(是否有机组投运, 机组电压设定值)`` 两个 ``[nb]`` 数组。

    没有机组的母线电压设定记为 0，由调用方按母线类型解释。
    """

    gen = problem.gen
    online = np.zeros(problem.nb, dtype=bool)
    setpoint = np.zeros(problem.nb, dtype=np.float64)
    for gen_index in np.flatnonzero(gen[:, GEN_STATUS] > 0):
        bus_index = int(gen[gen_index, GEN_BUS])
        online[bus_index] = True
        setpoint[bus_index] = gen[gen_index, VG]
    return online, setpoint


def expected_flat_voltage(problem: "PowerFlowProblem") -> np.ndarray:
    """独立推导平启动应有的电压幅值 ``[nb]``。

    PQ 母线为 1.0，PV 与平衡母线取发电机电压设定。这个函数给
    ``assert_flat_start`` 一个独立复算的基准。
    """

    bus_type = problem.bus[:, BUS_TYPE]
    _, setpoint = generator_bus_arrays(problem)
    return np.where(bus_type == PQ, 1.0, setpoint)


def assert_flat_start(problem: "PowerFlowProblem") -> None:
    """确认内部初值确实是平启动，而不是算例自带的那组已解好的状态。

    PYPOWER 的 ``case39`` 自带 ``VM``/``VA`` 就是一组解，初值处失配
    只有 2.9e-5 pu。若误用它当启动点，牛顿法一次迭代就收敛，
    ``iterations`` 这类元数据会全部失去意义。这个断言是防回归的闸门。
    """

    if not np.allclose(np.abs(problem.V0), expected_flat_voltage(problem), atol=1e-12):
        raise ValueError(
            "内部初值不是平启动：电压幅值与 (PQ=1.0, PV/平衡=VG) 的期望值不符。"
            "请用 prepare_case39(flat=True) 或 prepare_case(flat_start(ppc)) 构造算例。"
        )
    if not np.allclose(np.angle(problem.V0), 0.0, atol=1e-12):
        raise ValueError("内部初值不是平启动：电压相角不为零。")


def dense_ybus(problem: "PowerFlowProblem") -> np.ndarray:
    """把 ``Ybus`` 统一转成稠密复数数组。"""

    ybus = problem.Ybus
    return np.asarray(ybus.toarray() if hasattr(ybus, "toarray") else ybus, dtype=complex)


def _json_default(value: Any) -> Any:
    """让 json 能序列化 numpy 与 pandas 的类型。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 写出带缩进的 JSON。"""

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def write_checksums(root: Path, files: list[Path]) -> Path:
    """对给定文件逐个算 SHA-256，写成 ``<摘要>  <相对路径>`` 清单。"""

    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    target = root / CHECKSUMS_FILENAME
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def solve_with_pypower(
    problem: "PowerFlowProblem",
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[np.ndarray, bool]:
    """用 PYPOWER 求解同一个算例，返回**内部编号**下的复电压。

    PYPOWER 只能处理外部编号，所以先 ``int2ext`` 送进去，拿到结果后
    再按 ``ppci["order"]["bus"]`` 映射回内部编号，才能和自研解逐母线对齐。
    """

    external = pad_branch_columns(int2ext(copy.deepcopy(problem.ppci)))
    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=tol, PF_MAX_IT=max_iter)
    solved, success = runpf(external, options)

    external_voltage = solved["bus"][:, VM] * np.exp(1j * np.deg2rad(solved["bus"][:, VA]))
    # 外部编号 → 复电压，再按 i2e 拉回内部行序。
    external_ids = solved["bus"][:, BUS_I].astype(int)
    voltage_of_number = {
        int(number): external_voltage[row]
        for row, number in enumerate(external_ids.tolist())
    }
    i2e = np.asarray(problem.ppci["order"]["bus"]["i2e"], dtype=int)
    internal_voltage = np.asarray(
        [voltage_of_number[int(number)] for number in i2e], dtype=complex
    )
    return internal_voltage, bool(success)


def cross_check_pypower(
    problem: "PowerFlowProblem",
    voltage: np.ndarray,
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[str, Any]:
    """把自研解与 PYPOWER 的解逐母线对比。

    返回最大电压幅值差、最大相角差（弧度）、网损差（MW），以及
    PYPOWER 自己是否收敛。
    """

    reference, success = solve_with_pypower(problem, tol=tol, max_iter=max_iter)
    voltage = np.asarray(voltage, dtype=complex)
    ybus = dense_ybus(problem)

    loss_custom = float((voltage * np.conj(ybus.dot(voltage))).real.sum()) * problem.base_mva
    loss_reference = (
        float((reference * np.conj(ybus.dot(reference))).real.sum()) * problem.base_mva
    )

    return {
        "pypower_converged": success,
        "max_dvm_pu": float(np.abs(np.abs(voltage) - np.abs(reference)).max()),
        "max_dva_rad": float(np.abs(np.angle(voltage) - np.angle(reference)).max()),
        "d_loss_mw": loss_custom - loss_reference,
    }


@dataclass(frozen=True)
class MaskConfig:
    """控制"哪些量测已知、哪些需要预测"。

    ``structural`` 是默认语义：它严格对应潮流计算的已知量——

    * PQ 母线：P、Q 已知，V 未知；
    * PV 母线：P、V 已知，Q 未知；
    * 平衡母线：V 已知，P、Q 未知。

    这样每条样本都是一个**信息完备、可解**的潮流问题。赛事如果只做
    标准潮流，这个掩码分布就够了。

    但赛题把"鲁棒性"列为评价维度，真实调度里也常有量测缺失。因此额外提供：

    * ``dropout``：在 structural 基础上，把一小部分已知量也藏起来；
    * ``extra``：把一小部分本来未知的量"量测到"，作为额外已知量；
    * ``mixed``：按权重在三种模式间随机选，训练时最稳健。

    注意：dropout 会让问题变得**欠定**，标签仍然是那次潮流解的真值。
    这类样本不是"错数据"，而是故意保留的困难样本；训练时可以用
    ``mask_mode`` 过滤或加权。
    """

    mode: str = "mixed"
    structural_weight: float = 0.70
    dropout_weight: float = 0.20
    extra_weight: float = 0.10
    dropout_prob: float = 0.10
    extra_prob: float = 0.10

    def __post_init__(self) -> None:
        allowed = {"structural", "dropout", "extra", "mixed"}
        if self.mode not in allowed:
            raise ValueError(f"mode 必须是 {sorted(allowed)} 之一，收到 {self.mode!r}")
        weights = np.asarray(
            [self.structural_weight, self.dropout_weight, self.extra_weight],
            dtype=float,
        )
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("三种 mask 模式的权重必须非负且总和为正")
        for name, value in (
            ("dropout_prob", self.dropout_prob),
            ("extra_prob", self.extra_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 [0, 1] 内")


@dataclass(frozen=True)
class CompetitionSample:
    """一条赛事格式样本。

    数组形状：节点字段为 ``[nb]``，支路字段为 ``[ne]``。
    输入 P/Q 是净注入；掩码为 1 的位置填 0.0 占位。
    """

    # 输入（已掩码）
    node_p_mw: np.ndarray
    node_q_mvar: np.ndarray
    node_vm_pu: np.ndarray
    node_p_mask: np.ndarray
    node_q_mask: np.ndarray
    node_v_mask: np.ndarray

    # 官方标签
    target_vm_pu: np.ndarray
    target_branch_dtheta_deg: np.ndarray
    target_pf_mw: np.ndarray
    target_pt_mw: np.ndarray
    target_qf_mvar: np.ndarray
    target_qt_mvar: np.ndarray

    # 辅助标签：不参与官方评测，用来做物理自检、可视化、可选辅助损失
    aux_va_deg: np.ndarray
    aux_net_p_mw: np.ndarray
    aux_net_q_mvar: np.ndarray

    # 元数据
    converged: bool
    iterations: int
    final_norm: float
    initial_norm: float
    total_load_mw: float
    loss_mw: float
    sigma: float
    mask_mode: int
    n_known_p: int
    n_known_q: int
    n_known_v: int
    sample_id: int
    topology_hash: str
    case_hash: str


def _hash_arrays(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        _hash_array(hasher, array)
    return hasher.hexdigest()[:16]


def competition_topology(problem: "PowerFlowProblem") -> dict[str, np.ndarray]:
    """把与运行方式无关的拓扑信息打包成一个 NPZ 字典。

    节点顺序就是 ``problem.bus`` 的内部 0 基顺序；``node_ids`` 给出它对应的
    原始母线号。模型看到的第 ``i`` 个节点、输出的第 ``i`` 行，都按这个顺序。
    """

    bus = problem.bus
    branch = problem.branch
    nb = problem.nb
    ne = branch.shape[0]

    # ``ext2int`` 之后，内部第 i 行对应外部母线号 i2e[i]。case39 恰好是 1..39，
    # 但这里按映射表取，不依赖"编号连续"这个巧合。
    node_ids = np.asarray(problem.ppci["order"]["bus"]["i2e"], dtype=np.int32)

    edge_index = np.vstack(
        [branch[:, F_BUS].astype(np.int32), branch[:, T_BUS].astype(np.int32)]
    )
    node_kv = bus[:, BASE_KV].astype(np.float64)

    # 关键单位约定：PYPOWER 的 BR_B 是整条支路的总充电电纳 B；
    # 赛题要的是 B/2，所以这里除以 2，且之后不再乘任何倍数。
    branch_features = np.column_stack(
        [
            branch[:, BR_R],
            branch[:, BR_X],
            branch[:, BR_B] / 2.0,
        ]
    ).astype(np.float64)

    raw_tap = branch[:, TAP].astype(np.float64)
    shift_deg = branch[:, SHIFT].astype(np.float64)
    # PYPOWER 用 TAP=0 表示"没有变压器"；有效变比按 1.0 计算。
    effective_tap = np.where(raw_tap == 0.0, 1.0, raw_tap)

    branch_type = np.full(ne, BRANCH_TYPE_LINE, dtype=np.int8)
    branch_type[raw_tap != 0.0] = BRANCH_TYPE_TRANSFORMER
    branch_type[shift_deg != 0.0] = BRANCH_TYPE_PHASE_SHIFTER

    return {
        "schema_version": np.asarray(COMPETITION_SCHEMA_VERSION),
        "base_mva": np.asarray(problem.base_mva, dtype=np.float64),
        "nb": np.asarray(nb, dtype=np.int32),
        "ne": np.asarray(ne, dtype=np.int32),
        "node_ids": node_ids,
        "edge_index": edge_index,
        "branch_ids": np.arange(ne, dtype=np.int32),
        "branch_features": branch_features,
        "branch_type": branch_type,
        "node_kv": node_kv,
        "from_kv": node_kv[edge_index[0]],
        "to_kv": node_kv[edge_index[1]],
        # 以下字段赛题原文没有明确要求，但物理复算需要，单独放在 aux_ 前缀下。
        "aux_tap": effective_tap,
        "aux_tap_raw": raw_tap,
        "aux_shift_deg": shift_deg,
        "aux_rate_a_mva": branch[:, RATE_A].astype(np.float64),
        "aux_status": branch[:, BR_STATUS].astype(np.int8),
        "aux_bus_type": bus[:, BUS_TYPE].astype(np.int8),
        "topology_hash": np.asarray(topology_hash(problem)),
    }


def structural_masks(bus_type: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回标准潮流的三个掩码，``1`` 表示需要预测。

    * P：平衡母线需要预测，PQ/PV 已知；
    * Q：平衡母线与 PV 母线需要预测，PQ 已知；
    * V：PQ 母线需要预测，PV/平衡已知。
    """

    bus_type = np.asarray(bus_type)
    p_mask = (bus_type == REF).astype(np.uint8)
    q_mask = ((bus_type == REF) | (bus_type == PV)).astype(np.uint8)
    v_mask = (bus_type == PQ).astype(np.uint8)
    return p_mask, q_mask, v_mask


def _choose_mask_mode(config: MaskConfig, rng: np.random.Generator) -> int:
    if config.mode == "structural":
        return MASK_MODE_STRUCTURAL
    if config.mode == "dropout":
        return MASK_MODE_DROPOUT
    if config.mode == "extra":
        return MASK_MODE_EXTRA
    weights = np.asarray(
        [config.structural_weight, config.dropout_weight, config.extra_weight],
        dtype=float,
    )
    weights /= weights.sum()
    return int(rng.choice([MASK_MODE_STRUCTURAL, MASK_MODE_DROPOUT, MASK_MODE_EXTRA], p=weights))


def build_masks(
    bus_type: np.ndarray,
    config: MaskConfig,
    *,
    sample_id: int,
    mask_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """生成一条样本的三通道掩码，并返回实际采用的模式码。"""

    p_mask, q_mask, v_mask = structural_masks(bus_type)
    rng = np.random.default_rng(np.random.SeedSequence([mask_seed, sample_id]))
    mode = _choose_mask_mode(config, rng)

    masks = [p_mask, q_mask, v_mask]
    if mode == MASK_MODE_DROPOUT:
        # 只藏"本来已知"的量；已经需要预测的保持不变。
        for index, mask in enumerate(masks):
            hide = (rng.random(mask.shape[0]) < config.dropout_prob) & (mask == MASK_KNOWN)
            masks[index] = np.where(hide, MASK_PREDICT, mask).astype(np.uint8)
    elif mode == MASK_MODE_EXTRA:
        # 只揭示"本来未知"的量；已经有值的保持不变。
        for index, mask in enumerate(masks):
            reveal = (rng.random(mask.shape[0]) < config.extra_prob) & (
                mask == MASK_PREDICT
            )
            masks[index] = np.where(reveal, MASK_KNOWN, mask).astype(np.uint8)

    return masks[0], masks[1], masks[2], mode


def label_competition_sample(
    problem: "PowerFlowProblem",
    *,
    sample_id: int,
    sigma: float = 1.0,
    mask_config: MaskConfig | None = None,
    master_seed: int = 0,
    mask_seed: int = 0,
    tol: float = 1e-10,
    max_iter: int = 40,
) -> CompetitionSample:
    """求解一个潮流问题，并转成赛事要求的输入/输出格式。

    所有标签都来自**同一个**收敛解：节点 Vm、支路相角差、支路首末端 P/Q
    彼此之间严格满足潮流方程。这样模型学到的不是四组互不相干的回归目标，
    而是一组物理上自洽的答案。
    """

    assert_flat_start(problem)
    result = solve_newton_raphson(problem, tol=tol, max_iter=max_iter)

    voltage = np.asarray(result.voltage, dtype=complex)
    vm = np.abs(voltage)
    va = np.angle(voltage)
    va_deg = np.degrees(va)
    ybus = dense_ybus(problem)

    # 真实净注入 S = V ∘ conj(Ybus V)，标幺值。
    net_injection_pu = voltage * np.conj(ybus.dot(voltage))
    net_p_mw = net_injection_pu.real * problem.base_mva
    net_q_mvar = net_injection_pu.imag * problem.base_mva

    bus_type = problem.bus[:, BUS_TYPE]
    pd_pu, _ = base_load_pu(problem)
    total_load_mw = float(pd_pu.sum()) * problem.base_mva
    loss_mw = float(net_injection_pu.real.sum()) * problem.base_mva

    mask_config = mask_config or MaskConfig(mode="structural")
    p_mask, q_mask, v_mask, mask_mode = build_masks(
        bus_type,
        mask_config,
        sample_id=sample_id,
        mask_seed=mask_seed,
    )

    # 基准算例里"计划注入"：PQ/PV 的 P 和 PQ 的 Q 直接来自 Sbus。
    # 平衡母线的 P/Q、PV 母线的 Q 不是计划值，只能在解出来之后才知道，
    # 所以被 extra 模式揭示时用真实值填充。
    s_spec = np.asarray(problem.Sbus, dtype=complex)
    _, vg_setpoint = generator_bus_arrays(problem)

    p_value_mw = np.where(bus_type == REF, net_p_mw, s_spec.real * problem.base_mva)
    q_value_mvar = np.where(
        (bus_type == REF) | (bus_type == PV),
        net_q_mvar,
        s_spec.imag * problem.base_mva,
    )
    v_value_pu = np.where(bus_type == PQ, vm, vg_setpoint)

    node_p_mw = np.where(p_mask == MASK_KNOWN, p_value_mw, 0.0).astype(np.float64)
    node_q_mvar = np.where(q_mask == MASK_KNOWN, q_value_mvar, 0.0).astype(np.float64)
    node_vm_pu = np.where(v_mask == MASK_KNOWN, v_value_pu, 0.0).astype(np.float64)

    # 支路潮流：Sf = V_from ∘ conj(Yf V)，St = V_to ∘ conj(Yt V)。
    f = np.asarray(problem.branch[:, F_BUS], dtype=int)
    t = np.asarray(problem.branch[:, T_BUS], dtype=int)
    current_from = problem.Yf.dot(voltage)
    current_to = problem.Yt.dot(voltage)
    sf_mva = voltage[f] * np.conj(current_from) * problem.base_mva
    st_mva = voltage[t] * np.conj(current_to) * problem.base_mva

    dtheta_deg = np.degrees(va[f] - va[t])

    return CompetitionSample(
        node_p_mw=node_p_mw,
        node_q_mvar=node_q_mvar,
        node_vm_pu=node_vm_pu,
        node_p_mask=p_mask,
        node_q_mask=q_mask,
        node_v_mask=v_mask,
        target_vm_pu=vm.astype(np.float64),
        target_branch_dtheta_deg=dtheta_deg.astype(np.float64),
        target_pf_mw=sf_mva.real.astype(np.float64),
        target_pt_mw=st_mva.real.astype(np.float64),
        target_qf_mvar=sf_mva.imag.astype(np.float64),
        target_qt_mvar=st_mva.imag.astype(np.float64),
        aux_va_deg=va_deg.astype(np.float64),
        aux_net_p_mw=net_p_mw.astype(np.float64),
        aux_net_q_mvar=net_q_mvar.astype(np.float64),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_norm=float(result.final_norm),
        initial_norm=float(result.initial_norm),
        total_load_mw=total_load_mw,
        loss_mw=loss_mw,
        sigma=float(sigma),
        mask_mode=int(mask_mode),
        n_known_p=int((p_mask == MASK_KNOWN).sum()),
        n_known_q=int((q_mask == MASK_KNOWN).sum()),
        n_known_v=int((v_mask == MASK_KNOWN).sum()),
        sample_id=int(sample_id),
        topology_hash=topology_hash(problem),
        case_hash=_hash_arrays(node_p_mw, node_q_mvar, node_vm_pu, p_mask, q_mask, v_mask),
    )


def stack_competition_samples(samples: list[CompetitionSample]) -> dict[str, np.ndarray]:
    """把一批样本堆成带样本首轴的数组字典。"""

    if not samples:
        raise ValueError("stack_competition_samples 至少需要一条样本")

    shard: dict[str, np.ndarray] = {}
    for field in SAMPLE_ARRAY_FIELDS:
        shard[field] = np.stack([np.asarray(getattr(s, field)) for s in samples], axis=0)

    shard["converged"] = np.asarray([bool(s.converged) for s in samples])
    shard["iterations"] = np.asarray([int(s.iterations) for s in samples], dtype=np.int16)
    shard["mask_mode"] = np.asarray([int(s.mask_mode) for s in samples], dtype=np.int8)
    for field in ("n_known_p", "n_known_q", "n_known_v"):
        shard[field] = np.asarray([int(getattr(s, field)) for s in samples], dtype=np.int16)
    shard["sample_id"] = np.asarray([int(s.sample_id) for s in samples], dtype=np.uint64)
    for field in ("topology_hash", "case_hash"):
        shard[field] = np.asarray([str(getattr(s, field)) for s in samples], dtype="<U64")
    for field in (
        "final_norm",
        "initial_norm",
        "total_load_mw",
        "loss_mw",
        "sigma",
    ):
        shard[field] = np.asarray([float(getattr(s, field)) for s in samples])
    return shard


def _stats(samples: list[CompetitionSample]) -> dict[str, Any]:
    """给出训练归一化常用的统计量。

    注意：数据集本身存的是物理量。归一化只是训练技巧，不是赛题格式的一部分；
    这份统计表是为了让使用者不必自己再扫一遍数据。
    """

    def describe(values: np.ndarray) -> dict[str, float]:
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "node_p_mw": describe(np.concatenate([s.aux_net_p_mw for s in samples])),
        "node_q_mvar": describe(np.concatenate([s.aux_net_q_mvar for s in samples])),
        "node_vm_pu": describe(np.concatenate([s.target_vm_pu for s in samples])),
        "branch_dtheta_deg": describe(
            np.concatenate([s.target_branch_dtheta_deg for s in samples])
        ),
        "branch_pf_mw": describe(np.concatenate([s.target_pf_mw for s in samples])),
        "branch_pt_mw": describe(np.concatenate([s.target_pt_mw for s in samples])),
        "branch_qf_mvar": describe(np.concatenate([s.target_qf_mvar for s in samples])),
        "branch_qt_mvar": describe(np.concatenate([s.target_qt_mvar for s in samples])),
        "note": "统计量仅用于训练归一化；落盘数据始终是物理量。",
    }


def write_competition_dataset(
    root: str | Path,
    topology: dict[str, np.ndarray],
    samples: list[CompetitionSample],
    *,
    config: dict[str, Any],
    shard_size: int = 500,
    topology_name: str = "case39_base",
) -> dict[str, Any]:
    """写出赛事对齐数据集：拓扑一份 + 样本分片 + 配置/清单/校验和/统计。"""

    if not samples:
        raise ValueError("样本为空，拒绝写出空数据集")
    if shard_size <= 0:
        raise ValueError("shard_size 必须为正")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    topology_path = root / TOPOLOGY_DIRNAME / f"{topology_name}.npz"
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(topology_path, **topology)

    shard_paths: list[Path] = []
    shard_counts: list[int] = []
    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        chunk = samples[start : start + shard_size]
        shard_path = root / SAMPLES_DIRNAME / f"shard_{shard_index:06d}.npz"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard_path, **stack_competition_samples(chunk))
        shard_paths.append(shard_path)
        shard_counts.append(len(chunk))

    full_config = {
        "schema_version": COMPETITION_SCHEMA_VERSION,
        "n_samples": len(samples),
        "shard_size": shard_size,
        "topology_name": topology_name,
        "topology_hash": str(topology["topology_hash"]),
        **config,
    }
    write_json(root / CONFIG_FILENAME, full_config)

    manifest = {
        "schema_version": COMPETITION_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "topology": {
            "name": topology_name,
            "hash": str(topology["topology_hash"]),
            "nb": int(topology["nb"]),
            "ne": int(topology["ne"]),
        },
        "shards": [
            {"path": path.relative_to(root).as_posix(), "count": count}
            for path, count in zip(shard_paths, shard_counts)
        ],
    }
    write_json(root / MANIFEST_FILENAME, manifest)
    write_json(root / "stats.json", _stats(samples))
    write_checksums(root, [topology_path, *shard_paths])
    return manifest


class CompetitionDataset:
    """只依赖 numpy 的赛事格式读取器。

    ``__getitem__`` 返回的字典已经按赛题的语义分好组：

    * ``node_features`` ``[N, 3]``：P/Q/V，列顺序就是赛题顺序；
    * ``node_masks`` ``[N, 3]``：0 已知、1 待预测；
    * ``branch_features`` ``[E, 3]``：R/X/B2；
    * ``targets``：官方要求输出的六个量；
    * ``aux``：辅助物理量与拓扑元数据，不参与官方评测；
    * ``meta``：样本级标量。
    """

    def __init__(self, root: str | Path, *, cache_shards: bool = False) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据集目录不存在：{self.root}")

        # 训练时随机打乱下标，如果只缓存一个分片，跨分片访问会反复解压。
        # ``cache_shards=True`` 把全部分片常驻内存；当前 2000 条数据很小。
        self._cache_shards = bool(cache_shards)
        self._shard_cache: dict[int, dict[str, np.ndarray]] = {}

        self.config = json.loads((self.root / CONFIG_FILENAME).read_text(encoding="utf-8"))
        self.manifest = json.loads(
            (self.root / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )

        topology_path = self.root / TOPOLOGY_DIRNAME / f"{self.config['topology_name']}.npz"
        with np.load(topology_path) as data:
            self._topology = {key: data[key] for key in data.files}

        self._shard_paths = [self.root / entry["path"] for entry in self.manifest["shards"]]
        counts = [int(entry["count"]) for entry in self.manifest["shards"]]
        self._shard_starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._total = int(self._shard_starts[-1])

        self._cache_index: int | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self._total

    def topology(self) -> dict[str, np.ndarray]:
        return self._topology

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._total
        if not 0 <= index < self._total:
            raise IndexError(f"样本下标越界：{index}（共 {self._total} 条）")
        shard_index = int(np.searchsorted(self._shard_starts, index, side="right") - 1)
        return shard_index, index - int(self._shard_starts[shard_index])

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cache_shards:
            if shard_index not in self._shard_cache:
                with np.load(self._shard_paths[shard_index]) as data:
                    self._shard_cache[shard_index] = {
                        key: data[key] for key in data.files
                    }
            return self._shard_cache[shard_index]

        if self._cache_index != shard_index:
            with np.load(self._shard_paths[shard_index]) as data:
                self._cache = {key: data[key] for key in data.files}
            self._cache_index = shard_index
        assert self._cache is not None
        return self._cache

    def preload(self) -> None:
        """把所有分片读入内存，适合训练时随机访问。"""

        self._cache_shards = True
        for shard_index in range(len(self._shard_paths)):
            self._load_shard(shard_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local = self._locate(index)
        shard = self._load_shard(shard_index)
        sample = {key: shard[key][local] for key in shard}

        node_features = np.stack(
            [sample["node_p_mw"], sample["node_q_mvar"], sample["node_vm_pu"]], axis=-1
        )
        node_masks = np.stack(
            [sample["node_p_mask"], sample["node_q_mask"], sample["node_v_mask"]], axis=-1
        )

        return {
            "node_features": node_features,
            "node_masks": node_masks,
            "node_kv": self._topology["node_kv"],
            "node_ids": self._topology["node_ids"],
            "edge_index": self._topology["edge_index"],
            "branch_features": self._topology["branch_features"],
            "branch_type": self._topology["branch_type"],
            "from_kv": self._topology["from_kv"],
            "to_kv": self._topology["to_kv"],
            "branch_ids": self._topology["branch_ids"],
            "targets": {
                "vm_pu": sample["target_vm_pu"],
                "dtheta_deg": sample["target_branch_dtheta_deg"],
                "pf_mw": sample["target_pf_mw"],
                "pt_mw": sample["target_pt_mw"],
                "qf_mvar": sample["target_qf_mvar"],
                "qt_mvar": sample["target_qt_mvar"],
            },
            "aux": {
                "va_deg": sample["aux_va_deg"],
                "net_p_mw": sample["aux_net_p_mw"],
                "net_q_mvar": sample["aux_net_q_mvar"],
                "tap": self._topology["aux_tap"],
                "tap_raw": self._topology["aux_tap_raw"],
                "shift_deg": self._topology["aux_shift_deg"],
                "rate_a_mva": self._topology["aux_rate_a_mva"],
                "status": self._topology["aux_status"],
                "bus_type": self._topology["aux_bus_type"],
            },
            "meta": {key: sample[key] for key in SAMPLE_SCALAR_FIELDS},
        }

    def metadata(self) -> dict[str, np.ndarray]:
        collected: dict[str, list[np.ndarray]] = {key: [] for key in SAMPLE_SCALAR_FIELDS}
        for shard_index in range(len(self._shard_paths)):
            shard = self._load_shard(shard_index)
            for key in SAMPLE_SCALAR_FIELDS:
                collected[key].append(shard[key])
        return {key: np.concatenate(values) for key, values in collected.items()}

    def iter_batches(
        self,
        batch_size: int = 64,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> Iterator[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正")

        indices = np.arange(self._total)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)

        for start in range(0, self._total, batch_size):
            chunk = indices[start : start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            items = [self[int(i)] for i in chunk]
            yield {
                "node_features": np.stack([item["node_features"] for item in items]),
                "node_masks": np.stack([item["node_masks"] for item in items]),
                "node_kv": np.stack([item["node_kv"] for item in items]),
                "node_ids": np.stack([item["node_ids"] for item in items]),
                "edge_index": np.stack([item["edge_index"] for item in items]),
                "branch_features": np.stack([item["branch_features"] for item in items]),
                "branch_type": np.stack([item["branch_type"] for item in items]),
                "from_kv": np.stack([item["from_kv"] for item in items]),
                "to_kv": np.stack([item["to_kv"] for item in items]),
                "branch_ids": np.stack([item["branch_ids"] for item in items]),
                "targets": {
                    key: np.stack([item["targets"][key] for item in items])
                    for key in items[0]["targets"]
                },
                "aux": {
                    key: np.stack([item["aux"][key] for item in items])
                    for key in items[0]["aux"]
                },
                "meta": {
                    key: np.asarray([item["meta"][key] for item in items])
                    for key in items[0]["meta"]
                },
            }
