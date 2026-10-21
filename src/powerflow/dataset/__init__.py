"""赛事对齐的潮流图数据集。

所有字段、掩码、打标和读写都在 ``competition`` 模块里；``sampler``
只负责负荷场景采样。两者合起来就是当前比赛场景的数据集实现。
"""

from .competition import (
    BRANCH_FEATURE_NAMES,
    BRANCH_TARGET_NAMES,
    CHECKSUMS_FILENAME,
    COMPETITION_SCHEMA_VERSION,
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    MASK_KNOWN,
    MASK_PREDICT,
    NODE_FEATURE_NAMES,
    NODE_MASK_NAMES,
    CompetitionDataset,
    CompetitionSample,
    MaskConfig,
    assert_flat_start,
    competition_topology,
    cross_check_pypower,
    label_competition_sample,
    stack_competition_samples,
    topology_hash,
    write_competition_dataset,
)
from .sampler import (
    LoadScenario,
    PerturbationConfig,
    apply_scenario,
    base_load_pu,
    load_capacity_sigma,
    q_scaled_mask,
    sample_load_scenario,
)

__all__ = [
    "BRANCH_FEATURE_NAMES",
    "BRANCH_TARGET_NAMES",
    "CHECKSUMS_FILENAME",
    "COMPETITION_SCHEMA_VERSION",
    "CONFIG_FILENAME",
    "CompetitionDataset",
    "CompetitionSample",
    "LoadScenario",
    "MANIFEST_FILENAME",
    "MASK_KNOWN",
    "MASK_PREDICT",
    "MaskConfig",
    "NODE_FEATURE_NAMES",
    "NODE_MASK_NAMES",
    "PerturbationConfig",
    "apply_scenario",
    "assert_flat_start",
    "base_load_pu",
    "competition_topology",
    "cross_check_pypower",
    "label_competition_sample",
    "load_capacity_sigma",
    "q_scaled_mask",
    "sample_load_scenario",
    "stack_competition_samples",
    "topology_hash",
    "write_competition_dataset",
]
