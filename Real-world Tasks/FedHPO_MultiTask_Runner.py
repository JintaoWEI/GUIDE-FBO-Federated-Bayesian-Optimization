"""
Multi-task FBO experiments over FedHPO-Bench surrogate tasks.

Each FBO agent owns exactly one FedHPO-Bench task.  Algorithms continue to
optimize in [0, 1]^d, while this runner decodes every query with the owning
task's ConfigSpace before surrogate evaluation.
"""

from __future__ import annotations

import importlib
import io
import math
import pickle
import random
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# The official archives loaded by this runner were produced by
# build_surrogate_model(..., key="val_acc"). Each pickle is single-output, so
# its prediction target cannot be changed with a runtime configuration field.
SURROGATE_TARGET = "val_acc"
OBJECTIVE_DIRECTION = "maximize"


# =============================================================================
# 1. Experiment and task-group configuration
# =============================================================================


@dataclass(frozen=True)
class TaskSpec:
    model: str
    dataset: str
    fedalgo: str
    name: str = ""

    @property
    def display_name(self) -> str:
        return self.name or f"{self.model}/{self.dataset}/{self.fedalgo}"


@dataclass(frozen=True)
class TaskGroupSpec:
    name: str
    tasks: Tuple[TaskSpec, ...]


def _make_tasks(items: Sequence[Tuple[str, str, str]]) -> Tuple[TaskSpec, ...]:
    return tuple(TaskSpec(model, dataset, fedalgo) for model, dataset, fedalgo in items)


TASK_GROUP_REGISTRY: Dict[str, TaskGroupSpec] = {
    "GCN_AVG": TaskGroupSpec(
        name="GCN_AVG",
        tasks=_make_tasks(
            (
                ("gcn", "cora", "avg"),
                ("gcn", "citeseer", "avg"),
                ("gcn", "pubmed", "avg"),
            )
        ),
    ),
}


@dataclass
class MultiTaskExperimentConfig:
    algorithms: Tuple[str, ...] = ("GUIDE",)
    task_groups: Tuple[str, ...] = ("GCN_AVG",)
    initial_samples_per_agent: int = 30
    max_iterations: int = 50
    num_experiments: int = 10
    seed_base: int = 0
    fidelity_round: Optional[int] = None
    sample_client_rate: float = 1.0
    device: str = "cpu"
    dtype: str = "float64"
    output_dir: str = "fedhpo_multitask_results"
    fedhpo_root: str = "../FederatedScope/FedHPO-Bench-ICLR23"
    fedhpo_data_root: str = "../Real-world Tasks Data"

    # Existing algorithms model noisy observations internally.
    observation_noise: float = 1e-4
    enable_algorithm_debug: bool = False
    # Selects the base acquisition used inside GUIDE. The external result
    # name becomes GUIDE-UCB, GUIDE-NEI, or GUIDE-TS accordingly.
    guide_acquisition_function: str = "UCB"

    @property
    def budget_per_task(self) -> int:
        return self.initial_samples_per_agent + self.max_iterations


@dataclass
class GUIDEConfig:
    M: int = 300
    ORF_NUM_SAMPLES: int = 500
    DPGMM_COVARIANCE_TYPE: str = "diag"
    LAMBDA_MAX: float = 1.0
    RMS_EUCLIDEAN_THRESHOLD: float = 0.05
    USE_ORF_FOR_TS: bool = False
    SERVER_DISTRIBUTION_NUM: int = 5
    FIXED_BETA: float = 2.0
    VISUALIZE_PROCESS: bool = False


CONFIG = MultiTaskExperimentConfig()
GUIDE_CONFIG = GUIDEConfig()


ALGORITHM_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "CGP-NEI": {
        "CNEI_MAX_GROUP_SIZE": 4,
        "CNEI_COM_ITR": 1,
        "CNEI_MAX_SAMPLE": 20,
        "CNEI_MIN_SAMPLE": 5,
        "CNEI_TAU": 1e-4,
        "CNEI_LCB_BETA": 2.0,
    },
    "CGP-TS": {
        "CTS_N_TS_SAMPLES": 1000,
        "CTS_MAX_GROUP_SIZE": 4,
        "CTS_COM_ITR": 1,
        "CTS_MAX_SAMPLE": 20,
        "CTS_MIN_SAMPLE": 5,
        "CTS_TAU": 1e-4,
        "CTS_LCB_C": 2.0,
    },
    "CGP-UCB": {
        "CUCB_MAX_GROUP_SIZE": 4,
        "CUCB_COM_ITR": 1,
        "CUCB_MAX_SAMPLE": 20,
        "CUCB_MIN_SAMPLE": 5,
        "CUCB_TAU": 1e-4,
        "CUCB_LCB_BETA": 2.0,
        "CUCB_UCB_BETA": 4.0,
    },
    "FMTBO": {
        "FMTBO_KT_AGG_PROB": 0.8,
        "FMTBO_GAMMA": 0.5,
        "FMTBO_ENABLE_KT": True,
        "FMTBO_ACQ_TYPE": "EI_w",
    },
    "FTS": {"FTS_N_TS_SAMPLES": 1000, "FTS_M": 500},
    "FTSDE": {"FTSDE_N_TS_SAMPLES": 1000, "FTSDE_M": 500, "FTSDE_P": 4},
    "NEI": {},
    "TS": {"TS_N_TS_SAMPLES": 1000},
    "UCB": {"UCB_BETA": 2.0},
}


# =============================================================================
# 2. Existing algorithm registry
# =============================================================================


@dataclass(frozen=True)
class AlgorithmSpec:
    module_name: str
    agent_class_name: str
    run_function_name: str
    extra_module_names: Tuple[str, ...] = ()


ALGORITHM_REGISTRY: Dict[str, AlgorithmSpec] = {
    "GUIDE": AlgorithmSpec(
        module_name="GUIDE_Main",
        agent_class_name="Agent",
        run_function_name="run_single_experiment",
        extra_module_names=(
            "GUIDE_Config",
            "GUIDE_Utils",
            "GUIDE_Orf",
            "GUIDE_Agent",
            "GUIDE_Server",
        ),
    ),
    "CGP-NEI": AlgorithmSpec("CGP_NEI", "CNEIAgent", "run_single_cnei_experiment"),
    "CGP-TS": AlgorithmSpec("CGP_TS", "CTSAgent", "run_single_cts_experiment"),
    "CGP-UCB": AlgorithmSpec("CGP_UCB", "CUCBAgent", "run_single_cucb_experiment"),
    "FMTBO": AlgorithmSpec("FMTBO", "FMTBOAgent", "run_single_experiment"),
    "FTS": AlgorithmSpec("FTS", "FTSAgent", "run_single_fts_experiment"),
    "FTSDE": AlgorithmSpec("FTSDE", "FTSDEAgent", "run_single_ftsde_experiment"),
    "NEI": AlgorithmSpec("NEI", "NEIAgent", "run_single_nei_experiment"),
    "TS": AlgorithmSpec("TS", "TSAgent", "run_single_ts_experiment"),
    "UCB": AlgorithmSpec("UCB", "UCBAgent", "run_single_ucb_experiment"),
}


# =============================================================================
# 3. Search-space encoding and surrogate compatibility loading
# =============================================================================


def _resolve_dtype(name: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(aliases)}") from exc


def _nearest_choice(value: float, choices: Sequence[Any]) -> Any:
    try:
        return min(choices, key=lambda item: abs(float(item) - value))
    except (TypeError, ValueError):
        index = int(round(np.clip(value, 0.0, 1.0) * (len(choices) - 1)))
        return choices[index]


@dataclass(frozen=True)
class HyperparameterMetadata:
    name: str
    surrogate_name: str
    kind: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    choices: Tuple[Any, ...] = ()
    log: bool = False


class SearchSpaceEncoder:
    """Decode a stable normalized representation into a task ConfigSpace."""

    def __init__(self, config_space: Any):
        self.metadata: List[HyperparameterMetadata] = []

        # Sorting by name unifies ConfigSpace insertion-order differences.
        hyperparameters = sorted(config_space.values(), key=lambda item: item.name)
        for hp in hyperparameters:
            class_name = type(hp).__name__
            surrogate_name = "batch_size" if hp.name == "batch" else hp.name
            if "Categorical" in class_name or "Ordinal" in class_name:
                metadata = HyperparameterMetadata(
                    name=hp.name,
                    surrogate_name=surrogate_name,
                    kind="categorical",
                    choices=tuple(hp.choices),
                )
            elif "Integer" in class_name:
                metadata = HyperparameterMetadata(
                    name=hp.name,
                    surrogate_name=surrogate_name,
                    kind="integer",
                    lower=float(hp.lower),
                    upper=float(hp.upper),
                    log=bool(getattr(hp, "log", False)),
                )
            elif "Float" in class_name:
                metadata = HyperparameterMetadata(
                    name=hp.name,
                    surrogate_name=surrogate_name,
                    kind="continuous",
                    lower=float(hp.lower),
                    upper=float(hp.upper),
                    log=bool(getattr(hp, "log", False)),
                )
            else:
                raise TypeError(f"Unsupported ConfigSpace hyperparameter type: {class_name}")
            self.metadata.append(metadata)

    @property
    def dimension(self) -> int:
        return len(self.metadata)

    @property
    def signature(self) -> Tuple[HyperparameterMetadata, ...]:
        return tuple(self.metadata)

    def decode(self, normalized_x: torch.Tensor) -> Dict[str, Any]:
        values = normalized_x.detach().cpu().reshape(-1).numpy()
        if len(values) != self.dimension:
            raise ValueError(
                f"Normalized input dimension {len(values)} does not match "
                f"search-space dimension {self.dimension}"
            )

        configuration: Dict[str, Any] = {}
        for unit_value, metadata in zip(values, self.metadata):
            unit_value = float(np.clip(unit_value, 0.0, 1.0))
            if metadata.kind == "categorical":
                index = min(
                    int(math.floor(unit_value * len(metadata.choices))),
                    len(metadata.choices) - 1,
                )
                decoded = metadata.choices[index]
            elif metadata.log:
                assert metadata.lower is not None and metadata.upper is not None
                decoded = math.exp(
                    math.log(metadata.lower)
                    + unit_value
                    * (math.log(metadata.upper) - math.log(metadata.lower))
                )
            else:
                assert metadata.lower is not None and metadata.upper is not None
                decoded = metadata.lower + unit_value * (
                    metadata.upper - metadata.lower
                )

            if metadata.kind == "integer":
                decoded = int(round(decoded))
            configuration[metadata.surrogate_name] = decoded
        return configuration

    def describe(self) -> List[Dict[str, Any]]:
        return [asdict(item) for item in self.metadata]


def _loads_sklearn_compatible(data: bytes) -> Any:
    """Load FedHPO random-forest pickles across sklearn tree dtype versions."""
    try:
        return pickle.loads(data)
    except ValueError as exc:
        if "node array from the pickle has an incompatible dtype" not in str(exc):
            raise

    from sklearn.tree import _tree

    class CompatibleTree(_tree.Tree):
        def __setstate__(self, state: Dict[str, Any]) -> None:
            nodes = state.get("nodes")
            if nodes is not None and "missing_go_to_left" not in nodes.dtype.names:
                converted = np.zeros(nodes.shape, dtype=_tree.NODE_DTYPE)
                for field_name in nodes.dtype.names:
                    converted[field_name] = nodes[field_name]
                state = dict(state)
                state["nodes"] = converted
            super().__setstate__(state)

    class CompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "sklearn.tree._tree" and name == "Tree":
                return CompatibleTree
            return super().find_class(module, name)

    model = CompatibleUnpickler(io.BytesIO(data)).load()
    for estimator in getattr(model, "estimators_", ()):
        if not hasattr(estimator, "monotonic_cst"):
            estimator.monotonic_cst = None
    return model


def _load_surrogate_assets(task_dir: Path) -> Tuple[List[Any], Dict[str, Any]]:
    model_paths = sorted(task_dir.glob("surrogate_model_*.pkl"))
    models = [_loads_sklearn_compatible(path.read_bytes()) for path in model_paths]
    metadata = _loads_sklearn_compatible((task_dir / "info.pkl").read_bytes())
    return models, metadata


# =============================================================================
# 4. One-task and multi-task FedHPO adapters
# =============================================================================


class FedHPOTaskAdapter:
    """Own all surrogate assets and decoding state for one FedHPO task."""

    def __init__(
        self,
        task: TaskSpec,
        task_id: int,
        data_root: Path,
        config: MultiTaskExperimentConfig,
    ):
        self.task = task
        self.task_id = task_id
        self.data_root = data_root
        self.config = config
        self.current_experiment_id = 0
        self.evaluation_count = 0
        self.evaluation_batches: List[List[float]] = []

        self.task_dir = (
            data_root
            / "surrogate_model"
            / task.model
            / task.dataset
            / task.fedalgo
        )
        self._validate_surrogate_files()

        from fedhpobench.config import get_cs

        config_space, fidelity_space = get_cs(
            task.dataset, task.model, "surrogate", task.fedalgo
        )
        if len(config_space) == 0:
            raise ValueError(f"FedHPO-Bench defines no search space for {task.display_name}")

        models, metadata = _load_surrogate_assets(self.task_dir)
        self.surrogate_models = models
        self.surrogate_metadata = metadata
        self.config_space = self._complete_config_space(
            config_space, metadata, get_cs
        )
        self.fidelity_space = fidelity_space
        self.encoder = SearchSpaceEncoder(self.config_space)
        self.configuration_keys = tuple(
            sorted(
                "batch_size" if name == "batch" else name
                for name in metadata["configuration_space"]
            )
        )
        self.fidelity_keys = tuple(metadata["fidelity_space"])
        # The official builder trains every surrogate with features ordered as
        # [sorted configuration fields..., sample_client, round]. info.pkl
        # stores sorted fidelity names and therefore does not preserve this
        # training feature order.
        self.surrogate_fidelity_input_order = ("sample_client", "round")
        self.fidelity = self._build_fidelity()
        self._validate_interface()

    def _validate_surrogate_files(self) -> None:
        info_path = self.task_dir / "info.pkl"
        model_paths = list(self.task_dir.glob("surrogate_model_*.pkl"))
        missing: List[str] = []
        if not info_path.is_file():
            missing.append(str(info_path))
        if not model_paths:
            missing.append(str(self.task_dir / "surrogate_model_*.pkl"))
        if missing:
            raise FileNotFoundError(
                f"Missing surrogate assets for task {self.task.display_name}. "
                f"Expected: {missing}. Download the official "
                f"fedhpob_{self.task.model}_surrogate.zip archive."
            )

    def _complete_config_space(
        self, config_space: Any, metadata: Dict[str, Any], get_cs: Any
    ) -> Any:
        expected = set(metadata["configuration_space"])
        present = set(config_space.get_hyperparameter_names())
        missing = expected - present
        if not missing:
            return config_space

        # Official LR/MLP surrogate spaces omit batch although the RF input
        # metadata contains it; the tabular ConfigSpace provides its choices.
        tabular_space, _ = get_cs(
            self.task.dataset, self.task.model, "tabular", self.task.fedalgo
        )
        tabular_names = set(tabular_space.get_hyperparameter_names())
        for name in sorted(missing):
            config_name = "batch" if name == "batch_size" else name
            if config_name not in tabular_names:
                raise RuntimeError(
                    f"Cannot reconstruct surrogate field {name!r} for "
                    f"{self.task.display_name} from the tabular ConfigSpace"
                )
            config_space.add_hyperparameter(
                deepcopy(tabular_space.get_hyperparameter(config_name))
            )
        return config_space

    def _build_fidelity(self) -> Dict[str, Any]:
        round_hp = self.fidelity_space.get_hyperparameter("round")
        round_choices = tuple(round_hp.choices)
        selected_round = (
            max(round_choices)
            if self.config.fidelity_round is None
            else _nearest_choice(self.config.fidelity_round, round_choices)
        )

        rate_hp = self.fidelity_space.get_hyperparameter("sample_rate")
        selected_rate = _nearest_choice(
            self.config.sample_client_rate, tuple(rate_hp.choices)
        )
        rate_key = (
            "sample_client"
            if "sample_client" in self.surrogate_metadata["fidelity_space"]
            else "sample_rate"
        )
        return {"round": selected_round, rate_key: selected_rate}

    def _validate_interface(self) -> None:
        if not self.surrogate_models:
            raise RuntimeError(f"No surrogate models loaded for {self.task.display_name}")
        encoded_names = {item.surrogate_name for item in self.encoder.metadata}
        expected_names = set(self.configuration_keys)
        if encoded_names != expected_names:
            raise RuntimeError(
                f"Configuration fields mismatch for {self.task.display_name}: "
                f"encoder={sorted(encoded_names)}, surrogate={sorted(expected_names)}"
            )
        expected_fidelity = set(self.fidelity_keys)
        if expected_fidelity != set(self.surrogate_fidelity_input_order):
            raise RuntimeError(
                f"Unsupported fidelity fields for {self.task.display_name}: "
                f"{sorted(expected_fidelity)}. Expected sample_client and round."
            )
        if set(self.fidelity) != expected_fidelity:
            raise RuntimeError(
                f"Fidelity fields mismatch for {self.task.display_name}: "
                f"runner={sorted(self.fidelity)}, surrogate={sorted(expected_fidelity)}"
            )

    def set_experiment(self, experiment_id: int) -> None:
        self.current_experiment_id = experiment_id
        self.evaluation_count = 0
        self.evaluation_batches = []

    def evaluate(self, normalized_x: torch.Tensor) -> torch.Tensor:
        if normalized_x.dim() == 1:
            normalized_x = normalized_x.unsqueeze(0)

        model_index = (
            self.config.seed_base + self.current_experiment_id
        ) % len(self.surrogate_models)
        surrogate_model = self.surrogate_models[model_index]
        scores: List[float] = []
        for row in normalized_x:
            configuration = self.encoder.decode(row)
            model_input = [configuration[key] for key in self.configuration_keys]
            model_input.extend(
                self.fidelity[key] for key in self.surrogate_fidelity_input_order
            )
            prediction = float(surrogate_model.predict([model_input])[0])
            scores.append(prediction)

        self.evaluation_count += len(scores)
        self.evaluation_batches.append(list(scores))
        return torch.tensor(
            scores,
            device=torch.device(self.config.device),
            dtype=_resolve_dtype(self.config.dtype),
        ).unsqueeze(-1)

    def metadata(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task.display_name,
            "configuration_space": self.encoder.describe(),
            "fidelity": dict(self.fidelity),
            "surrogate_target": SURROGATE_TARGET,
            "objective_direction": OBJECTIVE_DIRECTION,
            "surrogate_fidelity_input_order": self.surrogate_fidelity_input_order,
            "surrogate_model_count": len(self.surrogate_models),
        }


class MultiTaskFedHPOAdapter:
    """Route agent i exclusively to task i and enforce a shared encoding."""

    def __init__(
        self,
        task_group: TaskGroupSpec,
        config: MultiTaskExperimentConfig,
    ):
        self.task_group = task_group
        self.config = config
        self.runner_dir = Path(__file__).resolve().parent
        self.root = (self.runner_dir / config.fedhpo_root).resolve()
        self.data_root = (self.runner_dir / config.fedhpo_data_root).resolve()
        self.device = torch.device(config.device)
        self.dtype = _resolve_dtype(config.dtype)

        if not self.root.is_dir():
            raise FileNotFoundError(f"FedHPO-Bench root does not exist: {self.root}")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"FedHPO-Bench data root does not exist: {self.data_root}")
        if not task_group.tasks:
            raise ValueError(f"Task group {task_group.name!r} is empty")

        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        self.task_adapters = tuple(
            FedHPOTaskAdapter(task, task_id, self.data_root, config)
            for task_id, task in enumerate(task_group.tasks)
        )
        self._validate_search_space_compatibility()
        self.encoder = self.task_adapters[0].encoder

    @property
    def num_agents(self) -> int:
        return len(self.task_adapters)

    @property
    def bounds(self) -> torch.Tensor:
        return torch.tensor(
            [[0.0] * self.encoder.dimension, [1.0] * self.encoder.dimension],
            device=self.device,
            dtype=self.dtype,
        )

    def _validate_search_space_compatibility(self) -> None:
        reference = self.task_adapters[0]
        reference_signature = reference.encoder.signature
        incompatible: List[str] = []
        for adapter in self.task_adapters[1:]:
            if adapter.encoder.signature != reference_signature:
                incompatible.append(
                    f"{adapter.task.display_name}: {adapter.encoder.describe()}"
                )
        if incompatible:
            raise ValueError(
                f"Task group {self.task_group.name} has incompatible search spaces. "
                f"Reference {reference.task.display_name}: "
                f"{reference.encoder.describe()}; mismatches: {incompatible}"
            )

    def set_experiment(self, experiment_id: int) -> None:
        for adapter in self.task_adapters:
            adapter.set_experiment(experiment_id)

    def evaluate(self, agent_id: int, normalized_x: torch.Tensor) -> torch.Tensor:
        if not isinstance(agent_id, (int, np.integer)):
            raise TypeError(f"agent_id must be an integer task id, got {agent_id!r}")
        if not 0 <= int(agent_id) < self.num_agents:
            raise IndexError(
                f"agent_id/task_id {agent_id} outside [0, {self.num_agents - 1}]"
            )
        return self.task_adapters[int(agent_id)].evaluate(normalized_x)

    def evaluation_counts(self) -> Tuple[int, ...]:
        return tuple(adapter.evaluation_count for adapter in self.task_adapters)

    def validate_budget(self, algorithm: str, experiment_id: int) -> None:
        expected = self.config.budget_per_task
        mismatches = {
            adapter.task.display_name: adapter.evaluation_count
            for adapter in self.task_adapters
            if adapter.evaluation_count != expected
        }
        if mismatches:
            raise RuntimeError(
                f"{algorithm} experiment {experiment_id} violated the equal per-task "
                f"budget {expected}: {mismatches}"
            )

    def aggregate_task_histories(
        self, algorithm: str, experiment_id: int
    ) -> Tuple[List[float], List[float]]:
        """Rebuild output curves directly from per-task evaluations.

        Initialization must be one batch containing all initial points. Each
        subsequent BO iteration must issue exactly one query for every task.
        The returned histories are the arithmetic mean of task-local metrics.
        """
        expected_batch_sizes = [
            self.config.initial_samples_per_agent,
            *([1] * self.config.max_iterations),
        ]
        task_btv_histories: List[List[float]] = []
        task_instantaneous_histories: List[List[float]] = []

        for task_adapter in self.task_adapters:
            actual_batch_sizes = [
                len(batch) for batch in task_adapter.evaluation_batches
            ]
            if actual_batch_sizes != expected_batch_sizes:
                raise RuntimeError(
                    f"{algorithm} experiment {experiment_id} produced an invalid "
                    f"query schedule for task {task_adapter.task.display_name}: "
                    f"batch_sizes={actual_batch_sizes}, "
                    f"expected={expected_batch_sizes}"
                )

            best_so_far = -math.inf
            task_btv: List[float] = []
            task_instantaneous: List[float] = []
            for batch in task_adapter.evaluation_batches:
                best_so_far = max(best_so_far, max(batch))
                task_btv.append(float(best_so_far))
                task_instantaneous.append(float(batch[-1]))
            task_btv_histories.append(task_btv)
            task_instantaneous_histories.append(task_instantaneous)

        mean_btv = np.mean(np.asarray(task_btv_histories), axis=0)
        mean_instantaneous = np.mean(
            np.asarray(task_instantaneous_histories), axis=0
        )
        return mean_btv.tolist(), mean_instantaneous.tolist()


# =============================================================================
# 5. Runtime injection into the existing ten algorithms
# =============================================================================


@dataclass
class LoadedAlgorithm:
    name: str
    spec: AlgorithmSpec
    main_module: Any
    modules: Tuple[Any, ...]
    agent_class: type
    run_single_experiment: Any


def _load_algorithm(name: str) -> LoadedAlgorithm:
    try:
        spec = ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algorithm {name!r}; choose from {sorted(ALGORITHM_REGISTRY)}"
        ) from exc

    # Load the local algorithms when this runner is launched from any directory.
    project_root = Path(__file__).resolve().parent.parent
    algorithm_dir = project_root / ("GUIDE-FBO" if name == "GUIDE" else "Baselines")
    if not algorithm_dir.is_dir():
        raise FileNotFoundError(f"Algorithm source directory does not exist: {algorithm_dir}")
    algorithm_path = str(algorithm_dir)
    if algorithm_path not in sys.path:
        sys.path.insert(0, algorithm_path)

    extra_modules = tuple(importlib.import_module(item) for item in spec.extra_module_names)
    main_module = importlib.import_module(spec.module_name)
    modules = tuple(dict.fromkeys((*extra_modules, main_module)))
    agent_class = next(
        (
            getattr(module, spec.agent_class_name)
            for module in modules
            if hasattr(module, spec.agent_class_name)
        ),
        None,
    )
    if agent_class is None:
        raise AttributeError(f"{name} does not expose {spec.agent_class_name}")
    return LoadedAlgorithm(
        name=name,
        spec=spec,
        main_module=main_module,
        modules=modules,
        agent_class=agent_class,
        run_single_experiment=getattr(main_module, spec.run_function_name),
    )


def _set_if_present(module: Any, key: str, value: Any) -> None:
    if hasattr(module, key):
        setattr(module, key, value)


def _inject_common_configuration(
    loaded: LoadedAlgorithm,
    adapter: MultiTaskFedHPOAdapter,
    config: MultiTaskExperimentConfig,
) -> None:
    common = {
        "bounds": adapter.bounds,
        "DIM": adapter.encoder.dimension,
        # FBO agents are tasks, never internal FL clients.
        "N_AGENTS": adapter.num_agents,
        "INITIAL_SAMPLES": config.initial_samples_per_agent,
        "MAX_ITERATIONS": config.max_iterations,
        "NUM_EXPERIMENTS": config.num_experiments,
        "SEED_BASE": config.seed_base,
        "DEVICE": adapter.device,
        "DTYPE": adapter.dtype,
        "NORMALIZE_X": 1.0,
        "NORMALIZE_Y": 1.0,
        "NOISE_SE": config.observation_noise,
        "ENABLE_DEBUG_PRINT": config.enable_algorithm_debug,
    }
    for module in loaded.modules:
        for key, value in common.items():
            _set_if_present(module, key, value)


def _inject_algorithm_configuration(
    loaded: LoadedAlgorithm,
    adapter: MultiTaskFedHPOAdapter,
    config: MultiTaskExperimentConfig,
) -> None:
    parameters = (
        asdict(GUIDE_CONFIG)
        if loaded.name == "GUIDE"
        else ALGORITHM_PARAMETERS.get(loaded.name, {})
    )
    parameters = dict(parameters)
    if loaded.name == "GUIDE":
        # Override the legacy module default from the runner configuration.
        parameters["ACQ_FUNCTION"] = config.guide_acquisition_function.upper()
    if loaded.name == "FTSDE":
        # FTSDE defines four geometric sub-regions. Never request more
        # sub-regions than agents, otherwise its original integer grouping
        # computes a zero group size.
        parameters["FTSDE_P"] = min(int(parameters["FTSDE_P"]), adapter.num_agents)
    missing: List[str] = []
    for key, value in parameters.items():
        updated = False
        for module in loaded.modules:
            if hasattr(module, key):
                setattr(module, key, value)
                updated = True
        if not updated:
            missing.append(key)
    if missing:
        raise AttributeError(
            f"{loaded.name} configuration fields are absent from its modules: {missing}"
        )


def _patch_ftsde_partitioning(
    loaded: LoadedAlgorithm, adapter: MultiTaskFedHPOAdapter
) -> None:
    """Balance arbitrary task counts across FTSDE's fixed four sub-regions.

    The original implementation assumes N_AGENTS is divisible by FTSDE_P.
    Both supported group sizes (3 and 7) violate its default P=4 assumption.
    This runner-only patch preserves the four geometric regions while assigning
    every task exactly once and including every task in transfer weights.
    """
    if loaded.name != "FTSDE":
        return

    module = loaded.main_module

    def balanced_initialize_agents(self: Any, exp_id: int = 0) -> None:
        self.agents = []
        for agent_id in range(self.n_agents):
            region_index = min(
                agent_id * module.FTSDE_P // self.n_agents,
                module.FTSDE_P - 1,
            )
            agent = module.FTSDEAgent(
                agent_id, self.subbounds[region_index], exp_id
            )
            agent.generate_initial_data(exp_id)
            self.agents.append(agent)

    def balanced_col_sample(self: Any, subbounds: Sequence[torch.Tensor]) -> torch.Tensor:
        values = torch.tensor([], device=module.DEVICE, dtype=module.DTYPE)
        candidates = torch.tensor([], device=module.DEVICE, dtype=module.DTYPE)
        groups = np.array_split(np.arange(module.N_AGENTS), module.FTSDE_P)
        for region_index, group in enumerate(groups):
            weights = torch.zeros(
                module.N_AGENTS, device=module.DEVICE, dtype=module.DTYPE
            )
            weights[group.tolist()] = 1
            temperature = (region_index + 1) ** 2
            weights = torch.exp((16 * weights + 1) / temperature)
            weights = weights / weights.sum()
            sub_weights = weights @ self.borrow_weights
            value, candidate = module.optimize_acquisition(
                self.predict, subbounds[region_index], sub_weights.data
            )
            values = torch.cat([values, value.unsqueeze(0)])
            candidates = torch.cat([candidates, candidate.unsqueeze(0)])
        return candidates[values.argmax()].unsqueeze(0)

    module.FTSDEServer.initialize_agents = balanced_initialize_agents
    module.FTSDEAgent.col_sample = balanced_col_sample


def _patch_agent_observation(
    agent_class: type, adapter: MultiTaskFedHPOAdapter
) -> None:
    def benchmark_observation(self: Any, x: torch.Tensor) -> torch.Tensor:
        # This is the semantic boundary: agent_id is always the owning task_id.
        values = adapter.evaluate(self.agent_id, x)
        self.instantaneous_value = float(values[-1].item())
        self.btv_value = max(float(self.btv_value), float(values.max().item()))
        return values

    agent_class.observation = benchmark_observation

    # Existing initializers overwrite benchmark BTV with their synthetic truth().
    # Always call the unwrapped initializer, then restore metrics from local_y.
    original_attr = "_fedhpo_multitask_original_generate_initial_data"
    if not hasattr(agent_class, original_attr):
        setattr(agent_class, original_attr, agent_class.generate_initial_data)
    original_generate = getattr(agent_class, original_attr)

    def benchmark_generate_initial_data(self: Any, exp_id: int = 0) -> None:
        original_generate(self, exp_id)
        if self.local_y is not None and self.local_y.numel() > 0:
            self.btv_value = float(self.local_y.max().item())
            self.instantaneous_value = float(self.local_y[-1].item())

    agent_class.generate_initial_data = benchmark_generate_initial_data


def prepare_algorithm(
    name: str,
    adapter: MultiTaskFedHPOAdapter,
    config: MultiTaskExperimentConfig,
) -> LoadedAlgorithm:
    loaded = _load_algorithm(name)
    _inject_common_configuration(loaded, adapter, config)
    _inject_algorithm_configuration(loaded, adapter, config)
    _patch_ftsde_partitioning(loaded, adapter)
    _patch_agent_observation(loaded.agent_class, adapter)
    return loaded


# =============================================================================
# 6. Execution, validation, and result persistence
# =============================================================================


@dataclass
class ExperimentResult:
    algorithm: str
    task_group: str
    best_so_far: List[List[float]] = field(default_factory=list)
    instantaneous: List[List[float]] = field(default_factory=list)
    elapsed_seconds: List[float] = field(default_factory=list)


def validate_config(config: MultiTaskExperimentConfig) -> None:
    if not config.algorithms:
        raise ValueError("algorithms cannot be empty")
    unknown_algorithms = sorted(set(config.algorithms) - set(ALGORITHM_REGISTRY))
    if unknown_algorithms:
        raise ValueError(f"Unknown algorithms: {unknown_algorithms}")

    if not config.task_groups:
        raise ValueError("task_groups cannot be empty")
    unknown_groups = sorted(set(config.task_groups) - set(TASK_GROUP_REGISTRY))
    if unknown_groups:
        raise ValueError(f"Unknown task groups: {unknown_groups}")

    positive_fields = {
        "initial_samples_per_agent": config.initial_samples_per_agent,
        "max_iterations": config.max_iterations,
        "num_experiments": config.num_experiments,
    }
    invalid = {name: value for name, value in positive_fields.items() if value <= 0}
    if invalid:
        raise ValueError(f"These fields must be positive integers: {invalid}")
    if config.observation_noise < 0:
        raise ValueError("observation_noise cannot be negative")
    if not 0 < config.sample_client_rate <= 1:
        raise ValueError("sample_client_rate must be in (0, 1]")
    if GUIDE_CONFIG.DPGMM_COVARIANCE_TYPE not in {"diag", "full"}:
        raise ValueError("DPGMM_COVARIANCE_TYPE must be 'diag' or 'full'")
    if config.guide_acquisition_function.upper() not in {"UCB", "NEI", "TS"}:
        raise ValueError(
            "guide_acquisition_function must be UCB, NEI, or TS"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _result_algorithm_name(
    algorithm_name: str,
    config: MultiTaskExperimentConfig,
) -> str:
    """Return the externally visible name used in logs and result files."""
    if algorithm_name == "GUIDE":
        return f"GUIDE-{config.guide_acquisition_function.upper()}"
    return algorithm_name


def _validate_history(
    algorithm: str,
    task_group: str,
    experiment_id: int,
    history: Sequence[float],
    expected_length: int,
    metric_name: str,
    require_monotonic: bool = False,
) -> List[float]:
    values = [float(item) for item in history]
    if len(values) != expected_length:
        raise RuntimeError(
            f"{algorithm}/{task_group} experiment {experiment_id} {metric_name} "
            f"length is {len(values)}, expected {expected_length}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"{algorithm}/{task_group} experiment {experiment_id} "
            f"{metric_name} contains non-finite values"
        )
    if require_monotonic and any(
        later + 1e-12 < earlier for earlier, later in zip(values, values[1:])
    ):
        raise RuntimeError(
            f"{algorithm}/{task_group} experiment {experiment_id} best-so-far "
            "history is not monotonically non-decreasing"
        )
    return values


def _validate_algorithm_aggregation(
    algorithm: str,
    task_group: str,
    experiment_id: int,
    metric_name: str,
    algorithm_history: Sequence[float],
    adapter_history: Sequence[float],
) -> None:
    """Ensure algorithm-side logging agrees with adapter-owned task means."""
    if not np.allclose(
        np.asarray(algorithm_history, dtype=float),
        np.asarray(adapter_history, dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        max_difference = float(
            np.max(
                np.abs(
                    np.asarray(algorithm_history, dtype=float)
                    - np.asarray(adapter_history, dtype=float)
                )
            )
        )
        raise RuntimeError(
            f"{algorithm}/{task_group} experiment {experiment_id} returned "
            f"{metric_name} that is not mean-over-tasks; "
            f"maximum difference from adapter aggregation is {max_difference}"
        )


def run_algorithm(
    algorithm_name: str,
    adapter: MultiTaskFedHPOAdapter,
    config: MultiTaskExperimentConfig,
) -> ExperimentResult:
    loaded = prepare_algorithm(algorithm_name, adapter, config)
    result_algorithm_name = _result_algorithm_name(algorithm_name, config)
    result = ExperimentResult(
        algorithm=result_algorithm_name, task_group=adapter.task_group.name
    )
    expected_length = config.max_iterations + 1

    print(
        f"\n[Runner] ===== group={adapter.task_group.name}, "
        f"algorithm={result_algorithm_name} ====="
    )
    for experiment_id in range(config.num_experiments):
        adapter.set_experiment(experiment_id)
        seed = config.seed_base + experiment_id
        _seed_everything(seed)
        print(
            f"[Runner] experiment {experiment_id + 1}/{config.num_experiments} "
            f"(seed={seed})"
        )

        reported_btv, reported_instantaneous, elapsed = loaded.run_single_experiment(
            experiment_id
        )
        adapter.validate_budget(algorithm_name, experiment_id)
        reported_btv = _validate_history(
            algorithm_name,
            adapter.task_group.name,
            experiment_id,
            reported_btv,
            expected_length,
            "reported_best_so_far",
            require_monotonic=True,
        )
        reported_instantaneous = _validate_history(
            algorithm_name,
            adapter.task_group.name,
            experiment_id,
            reported_instantaneous,
            expected_length,
            "reported_instantaneous",
        )
        mean_btv, mean_instantaneous = adapter.aggregate_task_histories(
            algorithm_name, experiment_id
        )
        mean_btv = _validate_history(
            algorithm_name,
            adapter.task_group.name,
            experiment_id,
            mean_btv,
            expected_length,
            "mean_task_best_so_far",
            require_monotonic=True,
        )
        mean_instantaneous = _validate_history(
            algorithm_name,
            adapter.task_group.name,
            experiment_id,
            mean_instantaneous,
            expected_length,
            "mean_task_instantaneous",
        )
        _validate_algorithm_aggregation(
            algorithm_name,
            adapter.task_group.name,
            experiment_id,
            "best_so_far",
            reported_btv,
            mean_btv,
        )
        _validate_algorithm_aggregation(
            algorithm_name,
            adapter.task_group.name,
            experiment_id,
            "instantaneous",
            reported_instantaneous,
            mean_instantaneous,
        )
        # Persist only adapter-owned mean-over-task histories.
        result.best_so_far.append(mean_btv)
        result.instantaneous.append(mean_instantaneous)
        result.elapsed_seconds.append(float(elapsed))
    return result


def _safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def save_result(
    result: ExperimentResult,
    adapter: MultiTaskFedHPOAdapter,
    config: MultiTaskExperimentConfig,
) -> Tuple[Path, Path, Path]:
    output_root = (Path(__file__).resolve().parent / config.output_dir).resolve()

    # Group results by the number of initial samples per agent.
    sample_count_dir = output_root / f"N{config.initial_samples_per_agent}"
    output_dir = sample_count_dir / _safe_name(result.task_group)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "_".join(
        (
            _safe_name(result.task_group),
            _safe_name(result.algorithm),
            f"A{adapter.num_agents}",
            f"N{config.initial_samples_per_agent}",
            f"I{config.max_iterations}",
            f"E{config.num_experiments}",
        )
    )
    iterations = list(range(config.max_iterations + 1))
    best_df = pd.DataFrame({"iteration": iterations})
    instant_df = pd.DataFrame({"iteration": iterations})
    for index in range(config.num_experiments):
        best_df[f"exp_{index + 1}"] = result.best_so_far[index]
        instant_df[f"exp_{index + 1}"] = result.instantaneous[index]

    best_path = output_dir / f"{prefix}_btv.csv"
    instant_path = output_dir / f"{prefix}_instantaneous.csv"
    timing_path = output_dir / f"{prefix}_timing.txt"
    best_df.to_csv(best_path, index=False)
    instant_df.to_csv(instant_path, index=False)

    total_evaluations = (
        adapter.num_agents * config.budget_per_task * config.num_experiments
    )
    timing_lines = [
        f"Algorithm: {result.algorithm}",
        f"Task group: {result.task_group}",
        f"Num agents: {adapter.num_agents}",
        f"Initial samples per agent: {config.initial_samples_per_agent}",
        f"Max iterations: {config.max_iterations}",
        f"Num experiments: {config.num_experiments}",
        f"Budget per task: {config.budget_per_task}",
        f"Total function evaluations: {total_evaluations}",
        f"Surrogate target: {SURROGATE_TARGET} (fixed by official pickle)",
        f"Objective direction: {OBJECTIVE_DIRECTION}",
        "",
        "Elapsed time for each experiment (seconds):",
    ]
    timing_lines.extend(
        f"Experiment {index + 1}: {elapsed:.6f}"
        for index, elapsed in enumerate(result.elapsed_seconds)
    )
    timing_lines.extend(
        (
            "",
            f"Total elapsed time: {np.sum(result.elapsed_seconds):.6f} seconds",
        )
    )
    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    print(f"[Runner] BTV: {best_path}")
    print(f"[Runner] instantaneous: {instant_path}")
    print(f"[Runner] timing: {timing_path}")
    return best_path, instant_path, timing_path


def print_configuration(
    config: MultiTaskExperimentConfig, adapter: MultiTaskFedHPOAdapter
) -> None:
    print(f"\n[Runner] task group: {adapter.task_group.name}")
    print(f"[Runner] tasks/agents: {adapter.num_agents}")
    for task_id, task in enumerate(adapter.task_group.tasks):
        print(f"[Runner]   agent {task_id} -> {task.display_name}")
    print(f"[Runner] search dimension: {adapter.encoder.dimension}")
    print(
        f"[Runner] per-task budget: {config.initial_samples_per_agent} + "
        f"{config.max_iterations} = {config.budget_per_task}"
    )
    print(
        f"[Runner] surrogate target={SURROGATE_TARGET} "
        f"(fixed), direction={OBJECTIVE_DIRECTION}"
    )


def main() -> None:
    validate_config(CONFIG)
    for task_group_name in CONFIG.task_groups:
        task_group = TASK_GROUP_REGISTRY[task_group_name]
        adapter = MultiTaskFedHPOAdapter(task_group, CONFIG)
        print_configuration(CONFIG, adapter)
        for algorithm_name in CONFIG.algorithms:
            result = run_algorithm(algorithm_name, adapter, CONFIG)
            save_result(result, adapter, CONFIG)
    print("\n[Runner] all multi-task experiments completed")


if __name__ == "__main__":
    main()
