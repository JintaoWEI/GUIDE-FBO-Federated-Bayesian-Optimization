"""Real-training multi-agent FBO benchmark on the official UCI-HAR dataset.

The official train/test files are merged first. Subject ``i`` is then mapped to
FBO agent ``i - 1`` and split into a local 50/50 train/validation task. Every
objective call creates and trains a fresh 561-to-6 PyTorch logistic-regression
model; no lookup table, surrogate objective, objective cache, or warm start is
used here.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
import tempfile
import time
import urllib.request
import warnings
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

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

BENCHMARK_NAME = "UCI_HAR"
OBJECTIVE_DIRECTION = "maximize"
UCI_HAR_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
EXPECTED_FEATURES = 561
EXPECTED_CLASSES = 6
EXPECTED_SUBJECTS = 30

DEFAULT_ALGORITHMS = (
    "GUIDE-UCB",
    "GUIDE-NEI",
    "GUIDE-TS",
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


# =============================================================================
# 1. Experiment configuration
# =============================================================================


@dataclass
class UCIHARExperimentConfig:
    algorithms: Tuple[str, ...] = DEFAULT_ALGORITHMS
    initial_samples_per_agent: int = 30
    max_iterations: int = 50
    num_experiments: int = 10
    seed_base: int = 0
    split_seed_base: int = 0

    training_epochs: int = 100
    standardization_epsilon: float = 1e-8

    device: str = "cpu"
    dtype: str = "float64"
    observation_noise: float = 1e-4
    enable_algorithm_debug: bool = False

    data_dir: str = "../Real-world Tasks Data/UCI_HAR/UCI HAR Dataset"
    output_dir: str = "uci_har_multitask_results"
    task_group_name: str = BENCHMARK_NAME

    @property
    def budget_per_agent(self) -> int:
        return self.initial_samples_per_agent + self.max_iterations


# Use the current real-world runner settings without mutating its shared object.
UCI_HAR_GUIDE_CONFIG = deepcopy(GUIDE_CONFIG)
UCI_HAR_GUIDE_CONFIG.M = 350
UCI_HAR_GUIDE_CONFIG.LAMBDA_MAX = 1.0
UCI_HAR_GUIDE_CONFIG.RMS_EUCLIDEAN_THRESHOLD = 0.05
UCI_HAR_GUIDE_CONFIG.USE_ORF_FOR_TS = False

CONFIG = UCIHARExperimentConfig()


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


def _safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


# =============================================================================
# 2. Official UCI-HAR data loading and subject-local splits
# =============================================================================


@dataclass(frozen=True)
class UCIHARSubjectData:
    subject_id: int
    features: np.ndarray
    labels: np.ndarray


def _dataset_files(root: Path) -> Dict[str, Path]:
    return {
        "x_train": root / "train" / "X_train.txt",
        "y_train": root / "train" / "y_train.txt",
        "subject_train": root / "train" / "subject_train.txt",
        "x_test": root / "test" / "X_test.txt",
        "y_test": root / "test" / "y_test.txt",
        "subject_test": root / "test" / "subject_test.txt",
    }


def resolve_dataset_root(configured_path: Path) -> Path:
    candidates = (configured_path, configured_path / "UCI HAR Dataset")
    for candidate in candidates:
        if all(path.is_file() for path in _dataset_files(candidate).values()):
            return candidate.resolve()
    missing = [
        str(path)
        for path in _dataset_files(configured_path).values()
        if not path.is_file()
    ]
    raise FileNotFoundError(
        "UCI-HAR data files were not found. Expected the extracted official "
        f"dataset under {configured_path}. Missing examples: {missing[:3]}. "
        "Run this runner once with --download-data or follow its guide."
    )


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract an official archive after rejecting path traversal entries."""
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"Unsafe path in UCI archive: {member.filename}")
    archive.extractall(destination)


def download_uci_har(configured_path: Path) -> Path:
    """Download and extract the official UCI archive if data is absent."""
    try:
        return resolve_dataset_root(configured_path)
    except FileNotFoundError:
        pass

    if configured_path.exists() and any(configured_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty incomplete data directory: {configured_path}"
        )

    configured_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Runner] downloading official UCI-HAR archive: {UCI_HAR_DOWNLOAD_URL}")
    with tempfile.TemporaryDirectory(prefix="uci_har_download_") as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / "uci_har.zip"
        with urllib.request.urlopen(UCI_HAR_DOWNLOAD_URL) as response:
            with archive_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)

        extracted = temporary_dir / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted)

        # The current UCI download can contain either the dataset directly or
        # a nested legacy "UCI HAR Dataset.zip" archive.
        nested_archives = list(extracted.rglob("UCI HAR Dataset.zip"))
        for nested_archive in nested_archives:
            with zipfile.ZipFile(nested_archive) as archive:
                _safe_extract(archive, extracted)

        source_roots = [
            path.parent.parent
            for path in extracted.rglob("train/X_train.txt")
            if all(item.is_file() for item in _dataset_files(path.parent.parent).values())
        ]
        if len(source_roots) != 1:
            raise RuntimeError(
                "Could not identify a unique UCI HAR Dataset root in the official "
                f"archive; found {source_roots}"
            )
        shutil.copytree(source_roots[0], configured_path)

    root = resolve_dataset_root(configured_path)
    print(f"[Runner] official UCI-HAR data extracted to: {root}")
    return root


def load_uci_har_subjects(data_root: Path) -> Tuple[UCIHARSubjectData, ...]:
    """Merge official train/test partitions and group all rows by subject."""
    root = resolve_dataset_root(data_root)
    files = _dataset_files(root)

    x_train = np.loadtxt(files["x_train"], dtype=np.float32)
    y_train = np.loadtxt(files["y_train"], dtype=np.int64).reshape(-1)
    subject_train = np.loadtxt(
        files["subject_train"], dtype=np.int64
    ).reshape(-1)
    x_test = np.loadtxt(files["x_test"], dtype=np.float32)
    y_test = np.loadtxt(files["y_test"], dtype=np.int64).reshape(-1)
    subject_test = np.loadtxt(files["subject_test"], dtype=np.int64).reshape(-1)

    features = np.vstack((x_train, x_test))
    # UCI activity labels are 1..6; CrossEntropyLoss requires 0..5.
    labels = np.concatenate((y_train, y_test)) - 1
    subjects = np.concatenate((subject_train, subject_test))

    if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected merged feature shape (*, {EXPECTED_FEATURES}), got {features.shape}"
        )
    if not (len(features) == len(labels) == len(subjects)):
        raise ValueError("Merged UCI-HAR feature, label, and subject lengths differ")
    if set(np.unique(labels).tolist()) != set(range(EXPECTED_CLASSES)):
        raise ValueError(f"Expected activity labels 0..5, found {np.unique(labels)}")
    if set(np.unique(subjects).tolist()) != set(range(1, EXPECTED_SUBJECTS + 1)):
        raise ValueError(f"Expected subject ids 1..30, found {np.unique(subjects)}")

    tasks: List[UCIHARSubjectData] = []
    for subject_id in range(1, EXPECTED_SUBJECTS + 1):
        mask = subjects == subject_id
        subject_features = np.ascontiguousarray(features[mask], dtype=np.float32)
        subject_labels = np.ascontiguousarray(labels[mask], dtype=np.int64)
        if set(np.unique(subject_labels).tolist()) != set(range(EXPECTED_CLASSES)):
            raise ValueError(
                f"Subject {subject_id} does not contain all six activities: "
                f"{np.unique(subject_labels)}"
            )
        tasks.append(
            UCIHARSubjectData(
                subject_id=subject_id,
                features=subject_features,
                labels=subject_labels,
            )
        )
    return tuple(tasks)


# =============================================================================
# 3. Fresh PyTorch logistic-regression objective
# =============================================================================


class UCIHARTaskAdapter:
    """One subject-local objective owned by exactly one FBO agent."""

    def __init__(
        self,
        task: UCIHARSubjectData,
        config: UCIHARExperimentConfig,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ):
        self.task = task
        self.config = config
        self.output_device = output_device
        self.output_dtype = output_dtype
        self.current_experiment_id = 0
        self.evaluation_count = 0
        self.model_training_count = 0
        self.evaluation_batches: List[List[float]] = []
        self.normalized_batches: List[np.ndarray] = []
        self.decoded_batches: List[List[Tuple[int, float, float]]] = []
        self.split_signature = ""
        self.x_train = torch.empty(0, EXPECTED_FEATURES)
        self.y_train = torch.empty(0, dtype=torch.long)
        self.x_validation = torch.empty(0, EXPECTED_FEATURES)
        self.y_validation = torch.empty(0, dtype=torch.long)

    def prepare_experiment(self, experiment_id: int) -> None:
        self.current_experiment_id = experiment_id
        indices = np.arange(len(self.task.labels))
        split_seed = self.config.split_seed_base + self.config.seed_base + experiment_id
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=0.5,
            stratify=self.task.labels,
            random_state=split_seed,
        )

        train_x = self.task.features[train_indices]
        validation_x = self.task.features[validation_indices]
        train_y = self.task.labels[train_indices]
        validation_y = self.task.labels[validation_indices]

        mean_train = train_x.mean(axis=0, dtype=np.float64)
        std_train = train_x.std(axis=0, dtype=np.float64)
        std_train = np.maximum(std_train, self.config.standardization_epsilon)
        train_x = ((train_x - mean_train) / std_train).astype(np.float32)
        validation_x = ((validation_x - mean_train) / std_train).astype(np.float32)

        self.x_train = torch.from_numpy(np.ascontiguousarray(train_x))
        self.y_train = torch.from_numpy(np.ascontiguousarray(train_y)).long()
        self.x_validation = torch.from_numpy(np.ascontiguousarray(validation_x))
        self.y_validation = torch.from_numpy(
            np.ascontiguousarray(validation_y)
        ).long()

        signature_payload = np.concatenate((train_indices, [-1], validation_indices))
        self.split_signature = hashlib.sha256(signature_payload.tobytes()).hexdigest()
        self.evaluation_count = 0
        self.model_training_count = 0
        self.evaluation_batches = []
        self.normalized_batches = []
        self.decoded_batches = []

    @staticmethod
    def decode(normalized_x: Sequence[float]) -> Tuple[int, float, float]:
        values = np.asarray(normalized_x, dtype=float).reshape(-1)
        if len(values) != 3:
            raise ValueError(f"UCI-HAR search dimension is 3, got {len(values)}")
        values = np.clip(values, 0.0, 1.0)
        batch_size = int(round(20.0 + 40.0 * float(values[0])))
        l2_regularization = 10.0 ** (-6.0 + 6.0 * float(values[1]))
        learning_rate = 10.0 ** (-2.0 + float(values[2]))
        return batch_size, l2_regularization, learning_rate

    def _training_seed(self, normalized_x: Sequence[float]) -> int:
        point = np.asarray(normalized_x, dtype="<f8").reshape(-1)
        digest = hashlib.blake2b(
            point.tobytes(), digest_size=8, person=b"UCI-HAR"
        ).digest()
        point_seed = int.from_bytes(digest, byteorder="little", signed=False)
        base = (
            self.config.seed_base
            + self.current_experiment_id * 1_000_003
            + self.task.subject_id * 10_007
        )
        return int((base + point_seed) % (2**31 - 1))

    def _validation_accuracy(self, normalized_x: Sequence[float]) -> float:
        batch_size, l2_regularization, learning_rate = self.decode(normalized_x)
        training_seed = self._training_seed(normalized_x)

        # Isolate objective randomness from BO randomness. A new model and a new
        # optimizer are deliberately created for every single objective call.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(training_seed)
            permutation_generator = torch.Generator(device="cpu")
            permutation_generator.manual_seed(training_seed + 1)

            model = nn.Linear(EXPECTED_FEATURES, EXPECTED_CLASSES)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=l2_regularization,
            )

            model.train()
            sample_count = len(self.x_train)
            for _ in range(self.config.training_epochs):
                permutation = torch.randperm(
                    sample_count, generator=permutation_generator
                )
                for start in range(0, sample_count, batch_size):
                    indices = permutation[start : start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(self.x_train[indices])
                    loss = criterion(logits, self.y_train[indices])
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions = model(self.x_validation).argmax(dim=1)
                accuracy = (
                    predictions.eq(self.y_validation).float().mean().item()
                )

        self.model_training_count += 1
        return float(accuracy)

    def evaluate(self, normalized_x: torch.Tensor) -> torch.Tensor:
        if normalized_x.dim() == 1:
            normalized_x = normalized_x.unsqueeze(0)
        if normalized_x.dim() != 2 or normalized_x.shape[1] != 3:
            raise ValueError(
                f"Expected normalized query shape (*, 3), got {tuple(normalized_x.shape)}"
            )

        normalized_rows = normalized_x.detach().cpu().numpy()
        scores = [self._validation_accuracy(row) for row in normalized_rows]
        decoded = [self.decode(row) for row in normalized_rows]
        self.evaluation_count += len(scores)
        self.evaluation_batches.append(list(scores))
        self.normalized_batches.append(np.asarray(normalized_rows, dtype=float).copy())
        self.decoded_batches.append(decoded)
        return torch.tensor(
            scores,
            device=self.output_device,
            dtype=self.output_dtype,
        ).unsqueeze(-1)


class MultiTaskUCIHARAdapter:
    """Route agent ``i`` to UCI-HAR subject ``i + 1``."""

    def __init__(self, config: UCIHARExperimentConfig):
        self.config = config
        self.runner_dir = Path(__file__).resolve().parent
        configured_path = (self.runner_dir / config.data_dir).resolve()
        self.data_root = resolve_dataset_root(configured_path)
        self.device = torch.device(config.device)
        self.dtype = _resolve_dtype(config.dtype)
        self.dimension = 3
        self.tasks = load_uci_har_subjects(self.data_root)
        self.task_adapters = tuple(
            UCIHARTaskAdapter(task, config, self.device, self.dtype)
            for task in self.tasks
        )
        self._split_references: Dict[int, Tuple[str, ...]] = {}
        self._clhs_references: Dict[int, Tuple[np.ndarray, ...]] = {}

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
        for task_adapter in self.task_adapters:
            task_adapter.prepare_experiment(experiment_id)
        signatures = tuple(task.split_signature for task in self.task_adapters)
        reference = self._split_references.setdefault(experiment_id, signatures)
        if signatures != reference:
            raise RuntimeError(
                f"Experiment {experiment_id} subject splits changed between algorithms"
            )

    def evaluate(self, agent_id: int, normalized_x: torch.Tensor) -> torch.Tensor:
        if not isinstance(agent_id, (int, np.integer)):
            raise TypeError(f"agent_id must be an integer, got {agent_id!r}")
        if not 0 <= int(agent_id) < self.num_agents:
            raise IndexError(
                f"agent_id {agent_id} is outside [0, {self.num_agents - 1}]"
            )
        return self.task_adapters[int(agent_id)].evaluate(normalized_x)

    def validate_budget_and_fresh_training(
        self, algorithm: str, experiment_id: int
    ) -> None:
        expected = self.config.budget_per_agent
        budget_mismatches = {
            task.task.subject_id: task.evaluation_count
            for task in self.task_adapters
            if task.evaluation_count != expected
        }
        training_mismatches = {
            task.task.subject_id: (
                task.model_training_count,
                task.evaluation_count,
            )
            for task in self.task_adapters
            if task.model_training_count != task.evaluation_count
        }
        if budget_mismatches:
            raise RuntimeError(
                f"{algorithm} experiment {experiment_id} violated per-agent "
                f"budget {expected}: {budget_mismatches}"
            )
        if training_mismatches:
            raise RuntimeError(
                f"{algorithm} experiment {experiment_id} reused objective results: "
                f"{training_mismatches}"
            )

    def validate_common_clhs(self, algorithm: str, experiment_id: int) -> None:
        if algorithm == "FTSDE":
            return
        initial_batches = tuple(
            task.normalized_batches[0].copy() for task in self.task_adapters
        )
        reference = self._clhs_references.setdefault(experiment_id, initial_batches)
        for agent_id, (actual, expected) in enumerate(
            zip(initial_batches, reference)
        ):
            if not np.array_equal(actual, expected):
                raise RuntimeError(
                    f"{algorithm} experiment {experiment_id} agent {agent_id} "
                    "did not use the common CLHS initialization points"
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
                    f"{algorithm} experiment {experiment_id} subject "
                    f"{task_adapter.task.subject_id} query batches are "
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
            np.asarray(task_instantaneous_histories), axis=0
        )
        return mean_btv.tolist(), mean_instantaneous.tolist()

    def total_model_trainings(self) -> int:
        return sum(task.model_training_count for task in self.task_adapters)


# =============================================================================
# 4. Runtime injection into the existing algorithms
# =============================================================================


def _algorithm_parts(name: str) -> Tuple[str, Optional[str]]:
    normalized = name.upper()
    if normalized.startswith("GUIDE-"):
        acquisition = normalized.split("-", maxsplit=1)[1]
        if acquisition not in {"UCB", "NEI", "TS"}:
            raise ValueError(f"Unsupported GUIDE acquisition in {name!r}")
        return "GUIDE", acquisition
    if normalized == "GUIDE":
        return "GUIDE", "TS"
    return name, None


def _inject_common_configuration(
    loaded: LoadedAlgorithm,
    adapter: MultiTaskUCIHARAdapter,
    config: UCIHARExperimentConfig,
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
    acquisition: Optional[str],
    adapter: MultiTaskUCIHARAdapter,
) -> None:
    parameters = (
        vars(UCI_HAR_GUIDE_CONFIG)
        if loaded.name == "GUIDE"
        else ALGORITHM_PARAMETERS.get(loaded.name, {})
    )
    parameters = dict(parameters)
    if loaded.name == "GUIDE":
        parameters["ACQ_FUNCTION"] = acquisition
    if loaded.name == "FTSDE":
        parameters["FTSDE_P"] = min(
            int(parameters["FTSDE_P"]), adapter.num_agents
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


def _patch_common_clhs(
    loaded: LoadedAlgorithm,
    adapter: MultiTaskUCIHARAdapter,
    config: UCIHARExperimentConfig,
) -> None:
    """Guarantee identical CLHS points for every non-FTSDE algorithm."""
    if loaded.name == "FTSDE":
        return

    def common_equal_clhs(
        n_agents: int,
        initial_samples: int,
        exp_id: int = 0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        current_rng_state = torch.get_rng_state()
        torch.manual_seed(config.seed_base + exp_id * 10_000)
        agents_cubes: List[List[Optional[torch.Tensor]]] = [
            [None, None] for _ in range(n_agents)
        ]
        bounds = adapter.bounds
        for dimension in range(bounds.shape[1]):
            lower, upper = bounds[:, dimension]
            divisions = torch.linspace(
                lower,
                upper,
                n_agents * initial_samples + 1,
                device=adapter.device,
                dtype=adapter.dtype,
            )
            for offset, agent_id_tensor in enumerate(torch.randperm(n_agents)):
                agent_id = int(agent_id_tensor.item())
                permutation = torch.randperm(initial_samples, device=adapter.device)
                cube_length = (upper - lower) / (n_agents * initial_samples)
                starts = divisions[offset:-1:n_agents][permutation].unsqueeze(-1)
                lengths = cube_length.reshape(1)
                if agents_cubes[agent_id][0] is None:
                    agents_cubes[agent_id] = [starts, lengths]
                else:
                    agents_cubes[agent_id][0] = torch.hstack(
                        (agents_cubes[agent_id][0], starts)
                    )
                    agents_cubes[agent_id][1] = torch.hstack(
                        (agents_cubes[agent_id][1], lengths)
                    )
        torch.set_rng_state(current_rng_state)
        return [
            (cube[0], cube[1])
            for cube in agents_cubes
            if cube[0] is not None and cube[1] is not None
        ]

    for module in loaded.modules:
        if hasattr(module, "equal_CLHS"):
            setattr(module, "equal_CLHS", common_equal_clhs)


def _patch_agent_observation(
    agent_class: type,
    adapter: MultiTaskUCIHARAdapter,
) -> None:
    def uci_har_observation(self: Any, x: torch.Tensor) -> torch.Tensor:
        values = adapter.evaluate(self.agent_id, x)
        self.instantaneous_value = float(values[-1].item())
        self.btv_value = max(float(self.btv_value), float(values.max().item()))
        return values

    agent_class.observation = uci_har_observation

    # Existing initializers recompute BTV through their synthetic truth().
    original_attr = "_uci_har_original_generate_initial_data"
    if not hasattr(agent_class, original_attr):
        setattr(agent_class, original_attr, agent_class.generate_initial_data)
    original_generate = getattr(agent_class, original_attr)

    def uci_har_generate_initial_data(self: Any, exp_id: int = 0) -> None:
        original_generate(self, exp_id)
        if self.local_y is not None and self.local_y.numel() > 0:
            self.btv_value = float(self.local_y.max().item())
            self.instantaneous_value = float(self.local_y[-1].item())

    agent_class.generate_initial_data = uci_har_generate_initial_data


def prepare_algorithm(
    algorithm_name: str,
    adapter: MultiTaskUCIHARAdapter,
    config: UCIHARExperimentConfig,
) -> LoadedAlgorithm:
    base_name, acquisition = _algorithm_parts(algorithm_name)
    loaded = _load_algorithm(base_name)
    _inject_common_configuration(loaded, adapter, config)
    _inject_algorithm_configuration(loaded, acquisition, adapter)
    _patch_common_clhs(loaded, adapter, config)
    _patch_ftsde_partitioning(loaded, adapter)
    _patch_agent_observation(loaded.agent_class, adapter)
    return loaded


# =============================================================================
# 5. Execution, validation, and existing-compatible output
# =============================================================================


@dataclass
class UCIHARExperimentResult:
    algorithm: str
    best_so_far: List[List[float]] = field(default_factory=list)
    instantaneous: List[List[float]] = field(default_factory=list)
    elapsed_seconds: List[float] = field(default_factory=list)
    model_training_counts: List[int] = field(default_factory=list)


def validate_config(config: UCIHARExperimentConfig) -> None:
    if not config.algorithms:
        raise ValueError("algorithms cannot be empty")
    for algorithm in config.algorithms:
        base_name, _ = _algorithm_parts(algorithm)
        if base_name not in ALGORITHM_REGISTRY:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    for name, value in {
        "initial_samples_per_agent": config.initial_samples_per_agent,
        "max_iterations": config.max_iterations,
        "num_experiments": config.num_experiments,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if config.training_epochs != 100:
        raise ValueError("UCI-HAR logistic regression must use exactly 100 epochs")
    if config.device.lower() != "cpu":
        raise ValueError("This deterministic UCI-HAR runner currently requires device='cpu'")
    if config.observation_noise < 0:
        raise ValueError("observation_noise cannot be negative")
    if config.standardization_epsilon <= 0:
        raise ValueError("standardization_epsilon must be positive")
    lower = UCIHARTaskAdapter.decode((0.0, 0.0, 0.0))
    upper = UCIHARTaskAdapter.decode((1.0, 1.0, 1.0))
    expected_lower = (20, 1e-6, 1e-2)
    expected_upper = (60, 1.0, 1e-1)
    if lower != expected_lower or not (
        upper[0] == expected_upper[0]
        and math.isclose(upper[1], expected_upper[1])
        and math.isclose(upper[2], expected_upper[2])
    ):
        raise RuntimeError(f"UCI-HAR decode contract failed: {lower}, {upper}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _validate_history(
    algorithm: str,
    experiment_id: int,
    name: str,
    values: Sequence[float],
    expected_length: int,
    require_monotonic: bool = False,
) -> List[float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_length,):
        raise RuntimeError(
            f"{algorithm} experiment {experiment_id} {name} has shape "
            f"{array.shape}, expected {(expected_length,)}"
        )
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise RuntimeError(
            f"{algorithm} experiment {experiment_id} {name} is not valid accuracy"
        )
    if require_monotonic and np.any(np.diff(array) < -1e-12):
        raise RuntimeError(
            f"{algorithm} experiment {experiment_id} best-so-far decreased"
        )
    return array.tolist()


def _validate_algorithm_aggregation(
    algorithm: str,
    experiment_id: int,
    name: str,
    reported: Sequence[float],
    adapter_owned: Sequence[float],
) -> None:
    if not np.allclose(reported, adapter_owned, rtol=1e-7, atol=1e-9):
        difference = float(
            np.max(np.abs(np.asarray(reported) - np.asarray(adapter_owned)))
        )
        raise RuntimeError(
            f"{algorithm} experiment {experiment_id} reported {name} differs "
            f"from adapter aggregation by {difference}"
        )


def run_algorithm(
    algorithm_name: str,
    adapter: MultiTaskUCIHARAdapter,
    config: UCIHARExperimentConfig,
) -> UCIHARExperimentResult:
    loaded = prepare_algorithm(algorithm_name, adapter, config)
    result = UCIHARExperimentResult(algorithm=algorithm_name)
    expected_length = config.max_iterations + 1
    print(f"\n[Runner] ===== {BENCHMARK_NAME} / {algorithm_name} =====")

    for experiment_id in range(config.num_experiments):
        adapter.set_experiment(experiment_id)
        seed = config.seed_base + experiment_id
        _seed_everything(seed)
        print(
            f"[Runner] experiment {experiment_id + 1}/{config.num_experiments} "
            f"(seed={seed})"
        )
        reported_btv, reported_instantaneous, elapsed = (
            loaded.run_single_experiment(experiment_id)
        )
        adapter.validate_budget_and_fresh_training(algorithm_name, experiment_id)
        adapter.validate_common_clhs(algorithm_name, experiment_id)

        reported_btv = _validate_history(
            algorithm_name,
            experiment_id,
            "reported_btv",
            reported_btv,
            expected_length,
            require_monotonic=True,
        )
        reported_instantaneous = _validate_history(
            algorithm_name,
            experiment_id,
            "reported_instantaneous",
            reported_instantaneous,
            expected_length,
        )
        mean_btv, mean_instantaneous = adapter.aggregate_task_histories(
            algorithm_name, experiment_id
        )
        mean_btv = _validate_history(
            algorithm_name,
            experiment_id,
            "mean_task_btv",
            mean_btv,
            expected_length,
            require_monotonic=True,
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
        result.best_so_far.append(mean_btv)
        result.instantaneous.append(mean_instantaneous)
        result.elapsed_seconds.append(float(elapsed))
        result.model_training_counts.append(adapter.total_model_trainings())
    return result


def save_result(
    result: UCIHARExperimentResult,
    adapter: MultiTaskUCIHARAdapter,
    config: UCIHARExperimentConfig,
) -> Tuple[Path, Path, Path]:
    output_root = (Path(__file__).resolve().parent / config.output_dir).resolve()
    output_dir = output_root / f"Results_{BENCHMARK_NAME}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "_".join(
        (
            BENCHMARK_NAME,
            _safe_name(result.algorithm),
            f"A{adapter.num_agents}",
            f"N{config.initial_samples_per_agent}",
            f"I{config.max_iterations}",
            f"E{config.num_experiments}",
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
    timing_path = output_dir / f"{prefix}_timing.txt"
    btv_frame.to_csv(btv_path, index=False)
    instantaneous_frame.to_csv(instantaneous_path, index=False)

    total_evaluations = (
        adapter.num_agents * config.budget_per_agent * config.num_experiments
    )
    timing_lines = [
        f"Algorithm: {result.algorithm}",
        f"Benchmark: {BENCHMARK_NAME}",
        f"Num agents/subjects: {adapter.num_agents}",
        f"Initial samples per agent: {config.initial_samples_per_agent}",
        f"Max iterations: {config.max_iterations}",
        f"Num experiments: {config.num_experiments}",
        f"Budget per agent: {config.budget_per_agent}",
        f"Total objective evaluations: {total_evaluations}",
        "Objective: fresh nn.Linear(561, 6) validation accuracy",
        f"Training epochs per objective evaluation: {config.training_epochs}",
        f"Objective direction: {OBJECTIVE_DIRECTION}",
        "Local split: 50% train / 50% validation, stratified per subject",
        "Preprocessing: local-train mean/std only",
        "",
        "Elapsed time and fresh model trainings for each experiment:",
    ]
    timing_lines.extend(
        f"Experiment {index + 1}: {elapsed:.6f} seconds, "
        f"{result.model_training_counts[index]} model trainings"
        for index, elapsed in enumerate(result.elapsed_seconds)
    )
    timing_lines.extend(
        (
            "",
            f"Total elapsed time: {np.sum(result.elapsed_seconds):.6f} seconds",
            f"Verified fresh model trainings: {sum(result.model_training_counts)}",
        )
    )
    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    print(f"[Runner] BTV: {btv_path}")
    print(f"[Runner] instantaneous: {instantaneous_path}")
    print(f"[Runner] timing: {timing_path}")
    return btv_path, instantaneous_path, timing_path


def print_configuration(
    config: UCIHARExperimentConfig,
    adapter: MultiTaskUCIHARAdapter,
) -> None:
    print(f"\n[Runner] benchmark: {BENCHMARK_NAME}")
    print(f"[Runner] data root: {adapter.data_root}")
    print(f"[Runner] agents/subjects: {adapter.num_agents}")
    for agent_id, task in enumerate(adapter.tasks):
        print(
            f"[Runner]   agent {agent_id} -> subject {task.subject_id} "
            f"({len(task.labels)} samples)"
        )
    print(f"[Runner] model: nn.Linear({EXPECTED_FEATURES}, {EXPECTED_CLASSES})")
    print("[Runner] search space: [0, 1]^3 -> batch size, L2, learning rate")
    print(
        f"[Runner] per-agent budget: {config.initial_samples_per_agent} + "
        f"{config.max_iterations} = {config.budget_per_agent}"
    )
    print(f"[Runner] algorithms: {', '.join(config.algorithms)}")


def main(config: Optional[UCIHARExperimentConfig] = None) -> None:
    runtime_config = deepcopy(config if config is not None else CONFIG)
    validate_config(runtime_config)
    adapter = MultiTaskUCIHARAdapter(runtime_config)
    print_configuration(runtime_config, adapter)
    for algorithm_name in runtime_config.algorithms:
        result = run_algorithm(algorithm_name, adapter, runtime_config)
        save_result(result, adapter, runtime_config)
    print("\n[Runner] all UCI-HAR experiments completed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download and extract the official UCI-HAR archive if missing.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root relative to this runner, or an absolute path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    selected_config = deepcopy(CONFIG)
    if arguments.data_dir is not None:
        selected_config.data_dir = arguments.data_dir
    if arguments.download_data:
        selected_path = Path(selected_config.data_dir)
        if not selected_path.is_absolute():
            selected_path = Path(__file__).resolve().parent / selected_path
        download_uci_har(selected_path.resolve())
    main(selected_config)
