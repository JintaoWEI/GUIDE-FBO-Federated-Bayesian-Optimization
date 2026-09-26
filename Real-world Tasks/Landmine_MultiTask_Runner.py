"""
Multi-task FBO experiments on the 29-field landmine detection dataset.

Each BO agent corresponds to one landmine field. The optimization objective is
fixed three-fold validation ROC-AUC of an RBF-SVM; the held-out test half is
evaluated only after optimization finishes.
"""

from __future__ import annotations

import math
import pickle
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.io import loadmat

from FedHPO_MultiTask_Runner import (
    ALGORITHM_PARAMETERS,
    ALGORITHM_REGISTRY,
    GUIDE_CONFIG,
    LoadedAlgorithm,
    _load_algorithm,
    _patch_ftsde_partitioning,
    _set_if_present,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Experiment configuration
# =============================================================================


@dataclass
class LandmineExperimentConfig:
    algorithms: Tuple[str, ...] = (
        "GUIDE",
        #"FTS",
        #"FTSDE",
        #"FMTBO",
        #"NEI",
        #"TS",
        #"UCB",
        #"CGP-NEI",
        #"CGP-UCB",
        #"CGP-TS",
    )
    initial_samples_per_agent: int = 30
    max_iterations: int = 50
    num_experiments: int = 10
    seed_base: int = 0

    # The validation objective is fixed across algorithms and repetitions.
    cv_folds: int = 3
    cv_seed: int = 0

    # RBF-SVM search space. Unit-cube points are decoded logarithmically.
    c_range: Tuple[float, float] = (1e-4, 10.0)
    gamma_range: Tuple[float, float] = (1e-2, 10.0)

    device: str = "cpu"
    dtype: str = "float64"
    observation_noise: float = 1e-4
    enable_algorithm_debug: bool = False
    # Selects the base acquisition used inside GUIDE. The external result
    # name becomes GUIDE-UCB, GUIDE-NEI, or GUIDE-TS accordingly.
    guide_acquisition_function: str = "TS"

    data_path: str = "../Real-world Tasks Data/landmine/landmine_formated_data.pkl"
    raw_data_path: str = "../Real-world Tasks Data/landmine/LandmineData.mat"
    output_dir: str = "landmine_multitask_results"
    task_group_name: str = "Landmine29"

    @property
    def budget_per_task(self) -> int:
        return self.initial_samples_per_agent + self.max_iterations

GUIDE_CONFIG.M = 100
GUIDE_CONFIG.LAMBDA_MAX = 1.0
GUIDE_CONFIG.RMS_EUCLIDEAN_THRESHOLD = 0.03
GUIDE_CONFIG.USE_ORF_FOR_TS = True
CONFIG = LandmineExperimentConfig()


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
        raise ValueError(f"Unsupported dtype {name!r}: {sorted(aliases)}") from exc


# =============================================================================
# 2. Landmine data and per-task objective
# =============================================================================


@dataclass(frozen=True)
class LandmineTaskData:
    task_id: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_landmine_tasks(
    data_path: Path,
    raw_data_path: Path,
    cv_folds: int,
) -> Tuple[LandmineTaskData, ...]:
    if data_path.is_file():
        with data_path.open("rb") as handle:
            data = pickle.load(handle)
    elif raw_data_path.is_file():
        raw_data = loadmat(raw_data_path)
        features = raw_data["feature"][0]
        labels = raw_data["label"][0]
        all_x_train: List[np.ndarray] = []
        all_y_train: List[np.ndarray] = []
        all_x_test: List[np.ndarray] = []
        all_y_test: List[np.ndarray] = []
        for feature, label in zip(features, labels):
            x_train, x_test, y_train, y_test = train_test_split(
                feature,
                label,
                test_size=0.5,
                stratify=label,
                random_state=0,
            )
            all_x_train.append(x_train)
            all_y_train.append(y_train)
            all_x_test.append(x_test)
            all_y_test.append(y_test)
        data = {
            "all_X_train": all_x_train,
            "all_Y_train": all_y_train,
            "all_X_test": all_x_test,
            "all_Y_test": all_y_test,
        }
    else:
        raise FileNotFoundError(
            "Neither the formatted Landmine pickle nor the raw MAT file exists: "
            f"{data_path}, {raw_data_path}"
        )

    required = {"all_X_train", "all_Y_train", "all_X_test", "all_Y_test"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"Landmine pickle is missing keys: {sorted(missing)}")

    lengths = {key: len(data[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Landmine task-list lengths differ: {lengths}")
    if next(iter(lengths.values())) != 29:
        raise ValueError(f"Expected 29 landmine fields, found {lengths}")

    tasks: List[LandmineTaskData] = []
    for task_id in range(29):
        x_train = np.asarray(data["all_X_train"][task_id], dtype=float)
        y_train = np.asarray(data["all_Y_train"][task_id]).reshape(-1)
        x_test = np.asarray(data["all_X_test"][task_id], dtype=float)
        y_test = np.asarray(data["all_Y_test"][task_id]).reshape(-1)

        if x_train.ndim != 2 or x_test.ndim != 2:
            raise ValueError(f"Task {task_id} features must be two-dimensional")
        if x_train.shape[1] != 9 or x_test.shape[1] != 9:
            raise ValueError(
                f"Task {task_id} must have 9 features, got "
                f"{x_train.shape[1]} and {x_test.shape[1]}"
            )
        if len(x_train) != len(y_train) or len(x_test) != len(y_test):
            raise ValueError(f"Task {task_id} feature/label lengths differ")
        if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
            raise ValueError(f"Task {task_id} is not a binary classification task")
        if int(np.bincount(y_train.astype(int)).min()) < cv_folds:
            raise ValueError(
                f"Task {task_id} has too few samples in one class for "
                f"{cv_folds}-fold stratified CV"
            )

        tasks.append(
            LandmineTaskData(
                task_id=task_id,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
        )
    return tuple(tasks)


class LandmineTaskAdapter:
    """Deterministic validation objective for one landmine field."""

    def __init__(
        self,
        task: LandmineTaskData,
        config: LandmineExperimentConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.task = task
        self.config = config
        self.device = device
        self.dtype = dtype
        self.evaluation_count = 0
        self.evaluation_batches: List[List[float]] = []
        self.normalized_batches: List[np.ndarray] = []

        splitter = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.cv_seed,
        )
        self.cv_splits = tuple(
            (train_index, validation_index)
            for train_index, validation_index in splitter.split(
                task.x_train, task.y_train
            )
        )

    def reset(self) -> None:
        self.evaluation_count = 0
        self.evaluation_batches = []
        self.normalized_batches = []

    @staticmethod
    def _log_decode(unit_value: float, bounds: Tuple[float, float]) -> float:
        unit_value = float(np.clip(unit_value, 0.0, 1.0))
        lower, upper = bounds
        return math.exp(
            math.log(lower) + unit_value * (math.log(upper) - math.log(lower))
        )

    def decode(self, normalized_x: Sequence[float]) -> Tuple[float, float]:
        values = np.asarray(normalized_x, dtype=float).reshape(-1)
        if len(values) != 2:
            raise ValueError(f"Landmine search dimension is 2, got {len(values)}")
        c_value = self._log_decode(values[0], self.config.c_range)
        gamma_value = self._log_decode(values[1], self.config.gamma_range)
        return c_value, gamma_value

    @staticmethod
    def _make_estimator(c_value: float, gamma_value: float) -> Any:
        # Scaling is fitted inside every CV fold and therefore does not leak
        # validation information into training.
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=c_value,
                gamma=gamma_value,
                probability=False,
            ),
        )

    def _validation_auc(self, normalized_x: Sequence[float]) -> float:
        c_value, gamma_value = self.decode(normalized_x)
        fold_scores: List[float] = []
        for train_index, validation_index in self.cv_splits:
            estimator = self._make_estimator(c_value, gamma_value)
            estimator.fit(
                self.task.x_train[train_index],
                self.task.y_train[train_index],
            )
            decision = estimator.decision_function(
                self.task.x_train[validation_index]
            )
            fold_scores.append(
                float(
                    roc_auc_score(
                        self.task.y_train[validation_index],
                        decision,
                    )
                )
            )

        return float(np.mean(fold_scores))

    def evaluate(self, normalized_x: torch.Tensor) -> torch.Tensor:
        if normalized_x.dim() == 1:
            normalized_x = normalized_x.unsqueeze(0)

        normalized_rows = normalized_x.detach().cpu().numpy()
        scores = [self._validation_auc(row) for row in normalized_rows]
        self.evaluation_count += len(scores)
        self.evaluation_batches.append(list(scores))
        self.normalized_batches.append(np.asarray(normalized_rows, dtype=float).copy())
        return torch.tensor(scores, device=self.device, dtype=self.dtype).unsqueeze(-1)

    def best_normalized_configuration(self) -> Tuple[np.ndarray, float]:
        if not self.evaluation_batches:
            raise RuntimeError(f"Task {self.task.task_id} has no observations")
        all_scores = np.concatenate(
            [np.asarray(batch, dtype=float) for batch in self.evaluation_batches]
        )
        all_x = np.vstack(self.normalized_batches)
        best_index = int(np.argmax(all_scores))
        return all_x[best_index].copy(), float(all_scores[best_index])

    def evaluate_held_out_test(self) -> Dict[str, float]:
        best_x, best_validation_auc = self.best_normalized_configuration()
        c_value, gamma_value = self.decode(best_x)
        estimator = self._make_estimator(c_value, gamma_value)
        estimator.fit(self.task.x_train, self.task.y_train)
        decision = estimator.decision_function(self.task.x_test)
        test_auc = float(roc_auc_score(self.task.y_test, decision))
        return {
            "task_id": float(self.task.task_id),
            "c": c_value,
            "gamma": gamma_value,
            "validation_auc": best_validation_auc,
            "test_auc": test_auc,
        }


class MultiTaskLandmineAdapter:
    """Route BO agent i exclusively to landmine field i."""

    def __init__(self, config: LandmineExperimentConfig):
        self.config = config
        self.runner_dir = Path(__file__).resolve().parent
        data_path = (self.runner_dir / config.data_path).resolve()
        raw_data_path = (self.runner_dir / config.raw_data_path).resolve()
        self.device = torch.device(config.device)
        self.dtype = _resolve_dtype(config.dtype)
        self.dimension = 2
        self.tasks = load_landmine_tasks(
            data_path,
            raw_data_path,
            config.cv_folds,
        )
        self.task_adapters = tuple(
            LandmineTaskAdapter(task, config, self.device, self.dtype)
            for task in self.tasks
        )

    @property
    def num_agents(self) -> int:
        return len(self.task_adapters)

    @property
    def bounds(self) -> torch.Tensor:
        return torch.tensor(
            [[0.0] * self.dimension, [1.0] * self.dimension],
            device=self.device,
            dtype=self.dtype,
        )

    def set_experiment(self, experiment_id: int) -> None:
        # The CV objective remains fixed. Repetitions vary only BO randomness.
        del experiment_id
        for task_adapter in self.task_adapters:
            task_adapter.reset()

    def evaluate(self, agent_id: int, normalized_x: torch.Tensor) -> torch.Tensor:
        if not isinstance(agent_id, (int, np.integer)):
            raise TypeError(f"agent_id must be an integer, got {agent_id!r}")
        if not 0 <= int(agent_id) < self.num_agents:
            raise IndexError(f"agent_id {agent_id} is outside [0, 28]")
        return self.task_adapters[int(agent_id)].evaluate(normalized_x)

    def evaluation_counts(self) -> Tuple[int, ...]:
        return tuple(task.evaluation_count for task in self.task_adapters)

    def validate_budget(self, algorithm: str, experiment_id: int) -> None:
        expected = self.config.budget_per_task
        mismatches = {
            task.task.task_id: task.evaluation_count
            for task in self.task_adapters
            if task.evaluation_count != expected
        }
        if mismatches:
            raise RuntimeError(
                f"{algorithm} experiment {experiment_id} violated per-task "
                f"budget {expected}: {mismatches}"
            )

    def aggregate_task_histories(
        self,
        algorithm: str,
        experiment_id: int,
    ) -> Tuple[List[float], List[float]]:
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
                    f"{algorithm} experiment {experiment_id} produced invalid "
                    f"query batches for task {task_adapter.task.task_id}: "
                    f"{actual_batch_sizes}, expected {expected_batch_sizes}"
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
            np.asarray(task_instantaneous_histories),
            axis=0,
        )
        return mean_btv.tolist(), mean_instantaneous.tolist()

    def evaluate_held_out_tests(self) -> List[Dict[str, float]]:
        return [
            task_adapter.evaluate_held_out_test()
            for task_adapter in self.task_adapters
        ]


# =============================================================================
# 3. Runtime injection into the existing ten algorithms
# =============================================================================


def _inject_common_configuration(
    loaded: LoadedAlgorithm,
    adapter: MultiTaskLandmineAdapter,
    config: LandmineExperimentConfig,
) -> None:
    common = {
        "bounds": adapter.bounds,
        "DIM": adapter.dimension,
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
    adapter: MultiTaskLandmineAdapter,
    config: LandmineExperimentConfig,
) -> None:
    parameters = (
        vars(GUIDE_CONFIG)
        if loaded.name == "GUIDE"
        else ALGORITHM_PARAMETERS.get(loaded.name, {})
    )
    parameters = dict(parameters)
    if loaded.name == "GUIDE":
        # Override the legacy module default from the runner configuration.
        parameters["ACQ_FUNCTION"] = config.guide_acquisition_function.upper()
    if loaded.name == "FTSDE":
        parameters["FTSDE_P"] = min(
            int(parameters["FTSDE_P"]),
            adapter.num_agents,
        )

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
            f"{loaded.name} configuration fields are absent: {missing}"
        )


def _patch_agent_observation(
    agent_class: type,
    adapter: MultiTaskLandmineAdapter,
) -> None:
    def landmine_observation(self: Any, x: torch.Tensor) -> torch.Tensor:
        values = adapter.evaluate(self.agent_id, x)
        self.instantaneous_value = float(values[-1].item())
        self.btv_value = max(float(self.btv_value), float(values.max().item()))
        return values

    agent_class.observation = landmine_observation

    original_attr = "_landmine_original_generate_initial_data"
    if not hasattr(agent_class, original_attr):
        setattr(agent_class, original_attr, agent_class.generate_initial_data)
    original_generate = getattr(agent_class, original_attr)

    def landmine_generate_initial_data(self: Any, exp_id: int = 0) -> None:
        original_generate(self, exp_id)
        if self.local_y is not None and self.local_y.numel() > 0:
            self.btv_value = float(self.local_y.max().item())
            self.instantaneous_value = float(self.local_y[-1].item())

    agent_class.generate_initial_data = landmine_generate_initial_data


def prepare_algorithm(
    name: str,
    adapter: MultiTaskLandmineAdapter,
    config: LandmineExperimentConfig,
) -> LoadedAlgorithm:
    loaded = _load_algorithm(name)
    _inject_common_configuration(loaded, adapter, config)
    _inject_algorithm_configuration(loaded, adapter, config)
    _patch_ftsde_partitioning(loaded, adapter)
    _patch_agent_observation(loaded.agent_class, adapter)
    return loaded


# =============================================================================
# 4. Execution, validation, and output
# =============================================================================


@dataclass
class LandmineExperimentResult:
    algorithm: str
    best_so_far: List[List[float]] = field(default_factory=list)
    instantaneous: List[List[float]] = field(default_factory=list)
    elapsed_seconds: List[float] = field(default_factory=list)
    test_results: List[List[Dict[str, float]]] = field(default_factory=list)


def validate_config(config: LandmineExperimentConfig) -> None:
    if not config.algorithms:
        raise ValueError("algorithms cannot be empty")
    unknown = sorted(set(config.algorithms) - set(ALGORITHM_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")

    positive = {
        "initial_samples_per_agent": config.initial_samples_per_agent,
        "max_iterations": config.max_iterations,
        "num_experiments": config.num_experiments,
        "cv_folds": config.cv_folds,
    }
    invalid = {key: value for key, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"These values must be positive: {invalid}")
    if config.cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    for name, bounds_value in (
        ("c_range", config.c_range),
        ("gamma_range", config.gamma_range),
    ):
        lower, upper = bounds_value
        if not 0 < lower < upper:
            raise ValueError(f"Invalid {name}: {bounds_value}")
    if config.observation_noise < 0:
        raise ValueError("observation_noise cannot be negative")
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
    config: LandmineExperimentConfig,
) -> str:
    """Return the externally visible name used in logs and result files."""
    if algorithm_name == "GUIDE":
        return f"GUIDE-{config.guide_acquisition_function.upper()}"
    return algorithm_name


def _validate_history(
    algorithm: str,
    experiment_id: int,
    name: str,
    history: Sequence[float],
    expected_length: int,
    monotonic: bool = False,
) -> List[float]:
    values = [float(value) for value in history]
    if len(values) != expected_length:
        raise RuntimeError(
            f"{algorithm} experiment {experiment_id} {name} has length "
            f"{len(values)}, expected {expected_length}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{algorithm} experiment {experiment_id} {name} is invalid")
    if monotonic and any(
        later + 1e-12 < earlier
        for earlier, later in zip(values, values[1:])
    ):
        raise RuntimeError(f"{algorithm} experiment {experiment_id} BTV decreased")
    return values


def _validate_algorithm_aggregation(
    algorithm: str,
    experiment_id: int,
    name: str,
    reported: Sequence[float],
    rebuilt: Sequence[float],
) -> None:
    if not np.allclose(reported, rebuilt, rtol=1e-10, atol=1e-12):
        difference = float(
            np.max(np.abs(np.asarray(reported) - np.asarray(rebuilt)))
        )
        raise RuntimeError(
            f"{algorithm} experiment {experiment_id} reported {name} is not "
            f"mean-over-tasks; max difference={difference}"
        )


def run_algorithm(
    algorithm_name: str,
    adapter: MultiTaskLandmineAdapter,
    config: LandmineExperimentConfig,
) -> LandmineExperimentResult:
    loaded = prepare_algorithm(algorithm_name, adapter, config)
    result_algorithm_name = _result_algorithm_name(algorithm_name, config)
    result = LandmineExperimentResult(algorithm=result_algorithm_name)
    expected_length = config.max_iterations + 1

    print(f"\n[Runner] ===== Landmine29 / {result_algorithm_name} =====")
    for experiment_id in range(config.num_experiments):
        seed = config.seed_base + experiment_id
        adapter.set_experiment(experiment_id)
        _seed_everything(seed)
        print(
            f"[Runner] experiment {experiment_id + 1}/"
            f"{config.num_experiments} (seed={seed})"
        )

        reported_btv, reported_instantaneous, elapsed = (
            loaded.run_single_experiment(experiment_id)
        )
        adapter.validate_budget(algorithm_name, experiment_id)
        reported_btv = _validate_history(
            algorithm_name,
            experiment_id,
            "reported_btv",
            reported_btv,
            expected_length,
            monotonic=True,
        )
        reported_instantaneous = _validate_history(
            algorithm_name,
            experiment_id,
            "reported_instantaneous",
            reported_instantaneous,
            expected_length,
        )

        mean_btv, mean_instantaneous = adapter.aggregate_task_histories(
            algorithm_name,
            experiment_id,
        )
        mean_btv = _validate_history(
            algorithm_name,
            experiment_id,
            "mean_task_btv",
            mean_btv,
            expected_length,
            monotonic=True,
        )
        mean_instantaneous = _validate_history(
            algorithm_name,
            experiment_id,
            "mean_task_instantaneous",
            mean_instantaneous,
            expected_length,
        )
        _validate_algorithm_aggregation(
            algorithm_name,
            experiment_id,
            "btv",
            reported_btv,
            mean_btv,
        )
        _validate_algorithm_aggregation(
            algorithm_name,
            experiment_id,
            "instantaneous",
            reported_instantaneous,
            mean_instantaneous,
        )

        test_results = adapter.evaluate_held_out_tests()
        mean_test_auc = float(
            np.mean([item["test_auc"] for item in test_results])
        )
        print(f"[Runner] held-out mean test ROC-AUC: {mean_test_auc:.6f}")

        result.best_so_far.append(mean_btv)
        result.instantaneous.append(mean_instantaneous)
        result.elapsed_seconds.append(float(elapsed))
        result.test_results.append(test_results)
    return result


def _safe_name(text: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in text
    )


def save_result(
    result: LandmineExperimentResult,
    adapter: MultiTaskLandmineAdapter,
    config: LandmineExperimentConfig,
) -> Tuple[Path, Path, Path, Path]:
    output_root = (Path(__file__).resolve().parent / config.output_dir).resolve()
    output_dir = output_root / f"Results_{_safe_name(config.task_group_name)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "_".join(
        (
            _safe_name(config.task_group_name),
            _safe_name(result.algorithm),
        )
    )
    iterations = list(range(config.max_iterations + 1))
    btv_frame = pd.DataFrame({"iteration": iterations})
    instantaneous_frame = pd.DataFrame({"iteration": iterations})
    for experiment_id in range(config.num_experiments):
        btv_frame[f"exp_{experiment_id + 1}"] = result.best_so_far[experiment_id]
        instantaneous_frame[f"exp_{experiment_id + 1}"] = (
            result.instantaneous[experiment_id]
        )

    btv_path = output_dir / f"{prefix}_btv.csv"
    instantaneous_path = output_dir / f"{prefix}_instantaneous.csv"
    test_path = output_dir / f"{prefix}_test_auc.csv"
    timing_path = output_dir / f"{prefix}_timing.txt"
    btv_frame.to_csv(btv_path, index=False)
    instantaneous_frame.to_csv(instantaneous_path, index=False)

    test_rows: List[Dict[str, Any]] = []
    for task_id in range(adapter.num_agents):
        row: Dict[str, Any] = {
            "task_id": task_id,
            "field_name": f"field_{task_id + 1}",
        }
        task_values: List[float] = []
        for experiment_id in range(config.num_experiments):
            item = result.test_results[experiment_id][task_id]
            row[f"exp_{experiment_id + 1}_test_auc"] = item["test_auc"]
            row[f"exp_{experiment_id + 1}_validation_auc"] = item[
                "validation_auc"
            ]
            row[f"exp_{experiment_id + 1}_C"] = item["c"]
            row[f"exp_{experiment_id + 1}_gamma"] = item["gamma"]
            task_values.append(item["test_auc"])
        row["mean_test_auc"] = float(np.mean(task_values))
        test_rows.append(row)

    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(test_path, index=False)

    total_evaluations = (
        adapter.num_agents * config.budget_per_task * config.num_experiments
    )
    experiment_test_means = [
        float(np.mean([item["test_auc"] for item in experiment]))
        for experiment in result.test_results
    ]
    timing_lines = [
        f"Algorithm: {result.algorithm}",
        f"Task group: {config.task_group_name}",
        f"Num agents/tasks: {adapter.num_agents}",
        f"Features per task: 9",
        f"Initial samples per agent: {config.initial_samples_per_agent}",
        f"Max iterations: {config.max_iterations}",
        f"Num experiments: {config.num_experiments}",
        f"Budget per task: {config.budget_per_task}",
        f"Total BO objective evaluations: {total_evaluations}",
        f"Objective: fixed {config.cv_folds}-fold validation ROC-AUC",
        f"CV seed: {config.cv_seed}",
        f"SVM: StandardScaler + RBF SVC(probability=False)",
        f"C range (log scale): {config.c_range}",
        f"Gamma range (log scale): {config.gamma_range}",
        "",
        "Elapsed time for each experiment (seconds):",
    ]
    timing_lines.extend(
        f"Experiment {index + 1}: {elapsed:.6f}"
        for index, elapsed in enumerate(result.elapsed_seconds)
    )
    timing_lines.append("")
    timing_lines.append("Mean held-out test ROC-AUC for each experiment:")
    timing_lines.extend(
        f"Experiment {index + 1}: {score:.6f}"
        for index, score in enumerate(experiment_test_means)
    )
    timing_lines.extend(
        (
            "",
            f"Total elapsed time: {np.sum(result.elapsed_seconds):.6f} seconds",
            (
                "Overall mean held-out test ROC-AUC: "
                f"{np.mean(experiment_test_means):.6f}"
            ),
        )
    )
    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    print(f"[Runner] BTV: {btv_path}")
    print(f"[Runner] instantaneous: {instantaneous_path}")
    print(f"[Runner] held-out test AUC: {test_path}")
    print(f"[Runner] timing: {timing_path}")
    return btv_path, instantaneous_path, test_path, timing_path


def print_configuration(
    config: LandmineExperimentConfig,
    adapter: MultiTaskLandmineAdapter,
) -> None:
    print("[Runner] Landmine multi-task experiment")
    print(f"[Runner] agents/tasks: {adapter.num_agents}")
    print("[Runner] agent i maps to landmine field i")
    print("[Runner] search space: normalized [0, 1]^2 -> log(C), log(gamma)")
    print(
        f"[Runner] validation objective: fixed {config.cv_folds}-fold "
        f"ROC-AUC (seed={config.cv_seed})"
    )
    print("[Runner] held-out test data is evaluated only after BO")
    print(
        f"[Runner] per-task budget: {config.initial_samples_per_agent} + "
        f"{config.max_iterations} = {config.budget_per_task}"
    )


def main() -> None:
    validate_config(CONFIG)
    adapter = MultiTaskLandmineAdapter(CONFIG)
    print_configuration(CONFIG, adapter)
    for algorithm_name in CONFIG.algorithms:
        result = run_algorithm(algorithm_name, adapter, CONFIG)
        save_result(result, adapter, CONFIG)
    print("\n[Runner] all Landmine29 experiments completed")


if __name__ == "__main__":
    main()
