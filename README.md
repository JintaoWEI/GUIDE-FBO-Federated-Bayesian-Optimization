# GUIDE-FBO: Guidance via Uncertainty Intervention and Distributional Exchange for Federated Bayesian Optimization

Code for GUIDE-FBO, nine comparison methods, and three real-world federated Bayesian optimization experiments.

**Paper:** https://doi.org/10.48550/arXiv.2609.35038

GUIDE-FBO fits local Gaussian-process models at participating agents, extracts beliefs about promising locations, and uses server-side aggregation to guide subsequent evaluations. This repository contains the synthetic benchmark implementation, sensitivity and ablation experiments, and runners for FedHPO-Bench GCN/FedAvg, UCI-HAR, and Landmine.

## Repository contents

| Path | Purpose |
| --- | --- |
| `GUIDE-FBO/GUIDE_Config.py` | Synthetic benchmark settings, heterogeneity, algorithm parameters, and output filenames. |
| `GUIDE-FBO/GUIDE_Main.py` | Main synthetic GUIDE-FBO experiment. |
| `GUIDE-FBO/GUIDE_Agent.py` | Local data, Gaussian-process fitting, belief extraction, and acquisition optimization. |
| `GUIDE-FBO/GUIDE_Server.py` | Belief aggregation, calibration, merging, and distribution. |
| `GUIDE-FBO/GUIDE_Utils.py` | Benchmark objectives, initialization, distributions, and guided models. |
| `GUIDE-FBO/GUIDE_Orf.py` | Orthogonal random features for Thompson sampling. |
| `GUIDE-FBO/GUIDE_Sensitivity_Ablation.py` | GUIDE-FBO sensitivity and ablation runs. |
| `Baselines/UCB.py` | Independent Gaussian-process upper confidence bound baseline. |
| `Baselines/NEI.py` | Independent noisy expected improvement baseline. |
| `Baselines/TS.py` | Independent Thompson sampling baseline. |
| `Baselines/FTS.py` | Federated Thompson sampling baseline. |
| `Baselines/FTSDE.py` | Federated Thompson sampling baseline with subregion assignments and random Fourier features. |
| `Baselines/FMTBO.py` | Federated Bayesian optimization baseline with weighted expected improvement and knowledge transfer. |
| `Baselines/CGP_UCB.py` | Collaborative Gaussian-process baseline with upper confidence bound acquisition. |
| `Baselines/CGP_NEI.py` | Collaborative Gaussian-process baseline with noisy expected improvement acquisition. |
| `Baselines/CGP_TS.py` | Collaborative Gaussian-process baseline with Thompson sampling acquisition. |
| `Real-world Tasks/FedHPO_MultiTask_Runner.py` | FedHPO-Bench GCN/FedAvg on Cora, Citeseer, and Pubmed; also provides shared algorithm-loading utilities. |
| `Real-world Tasks/UCI_HAR_MultiTask_Runner.py` | Subject-specific UCI-HAR classifier training. |
| `Real-world Tasks/Landmine_MultiTask_Runner.py` | Landmine RBF-SVM tuning and held-out test AUC. |
| `requirements.txt` | Package versions exported from a Windows environment. |

All ten synthetic implementations support the same 12 objectives: `ackley`, `levy`, `griewank`, `rastrigin`, `rosenbrock`, `sphere`, `weierstrass`, `ellipsoid`, `zakharov`, `michalewicz`, `powell`, and `styblinskitang`.

## Environment

Use a compatible Windows Python environment and install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is a 183-package environment snapshot, including BoTorch 0.9.5, PyTorch 2.7.1, and pinned public revisions of FederatedScope and FedHPO-Bench. It also includes Windows-only packages such as `pywin32` and `pywinpty`, so direct installation on Linux or macOS is not supported by this snapshot. The complete snapshot has not been independently installed and tested end to end. Module import checks were run separately on Python 3.10 with BoTorch 0.10.0; that check does not establish numerical equality with the paper's runs.

FedHPO additionally needs the source checkouts at the paths expected by its runner. The two packages are included as public Git dependencies in `requirements.txt`, but that installation does not create these project-relative checkout paths. From the repository root:

```bash
mkdir FederatedScope
git clone https://github.com/alibaba/FederatedScope.git FederatedScope/upstream
git -C FederatedScope/upstream checkout a653102c6b8d5421d2874077594df0b2e29401a1
git clone https://github.com/FedHPO-Bench/FedHPO-Bench-ICLR23.git FederatedScope/FedHPO-Bench-ICLR23
git -C FederatedScope/FedHPO-Bench-ICLR23 checkout 995a03d872a5e2fc28a06880239d703b5298696d
```

## Quick start: synthetic experiments

Run commands from the repository root. The default GUIDE-FBO configuration uses Ackley, 16 agents, mild heterogeneity, 10 random repetitions, and 50 optimization rounds:

```bash
python "GUIDE-FBO/GUIDE_Main.py"
```

Select a different objective, acquisition function, number of agents, budget, seed count, or heterogeneity in `GUIDE-FBO/GUIDE_Config.py`. The acquisition choices are `UCB`, `NEI`, and `TS`. The heterogeneity settings `(HETER_SHIFT, HETER_ROTATION)` documented by the code are homogeneous `(0.0, 0.0)`, mild `(0.05, 0.1)`, and severe `(0.3, 1.0)`.

Each baseline is a standalone script; for example:

```bash
python "Baselines/UCB.py"
python "Baselines/FTS.py"
python "Baselines/CGP_TS.py"
```

The nine baseline scripts are `UCB.py`, `NEI.py`, `TS.py`, `FTS.py`, `FTSDE.py`, `FMTBO.py`, `CGP_UCB.py`, `CGP_NEI.py`, and `CGP_TS.py`. Their default synthetic run selects Ellipsoid with severe heterogeneity. **Set the same objective, heterogeneity, observation noise, initial samples, agent count, evaluation budget, and seeds before comparing GUIDE-FBO and baseline curves.** Function-specific dimensions and normalization values are drawn from each script's `FUNCTION_CONFIGS`.

For sensitivity or ablation experiments, edit `RUN_SENSITIVITY`, `RUN_ABLATION`, the objective lists, and the value lists at the top of `GUIDE-FBO/GUIDE_Sensitivity_Ablation.py`, then run:

```bash
python "GUIDE-FBO/GUIDE_Sensitivity_Ablation.py"
```

The sensitivity runner sweeps `M`, `P`, `LAMBDA_MAX`, and `RMS_EUCLIDEAN_THRESHOLD`. Completed result groups are skipped unless `OVERWRITE_EXISTING=True`.

## Real-world data

Real-world runners resolve the following paths relative to `Real-world Tasks/`. These datasets and the two external source checkouts are **not bundled** with this repository.

```text
Real-world Tasks Data/
  UCI_HAR/UCI HAR Dataset/
    train/X_train.txt
    train/y_train.txt
    train/subject_train.txt
    test/X_test.txt
    test/y_test.txt
    test/subject_test.txt
  landmine/
    LandmineData.mat
  surrogate_model/gcn/
    cora/avg/info.pkl
    cora/avg/surrogate_model_*.pkl
    citeseer/avg/info.pkl
    citeseer/avg/surrogate_model_*.pkl
    pubmed/avg/info.pkl
    pubmed/avg/surrogate_model_*.pkl
```

| Task | Source and preparation | Run |
| --- | --- | --- |
| UCI-HAR | Download the [official UCI-HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and place the six listed files under `UCI_HAR/UCI HAR Dataset/`. The 30 subjects define separate tasks. | `python "Real-world Tasks/UCI_HAR_MultiTask_Runner.py"` |
| Landmine | Download [`LandmineData.mat`](https://www.cs.columbia.edu/~jebara/code/multisparse/LandmineData.mat) into `landmine/`. The runner rebuilds the 29 train/test splits with `random_state=0`. | `python "Real-world Tasks/Landmine_MultiTask_Runner.py"` |
| FedHPO GCN/FedAvg | Download the [official GCN surrogate archive](https://federatedscope.oss-cn-beijing.aliyuncs.com/fedhpob_gcn_surrogate.zip), then copy each `avg` directory's `info.pkl` and **all** `surrogate_model_*.pkl` files into the three listed paths. The task group is fixed to `GCN_AVG`. | `python "Real-world Tasks/FedHPO_MultiTask_Runner.py"` |

The verified Landmine source file has SHA-256 `2163ab20a80e33a8b65593c48cf283c540599dcb291105e63fa4af2ce24c2bea`. The runner's reconstruction from that file matched the prepared split across all train/test feature and label arrays for the 29 tasks.

The FedHPO runner reads the surrogate pickles directly and loads all models for Cora, Citeseer, and Pubmed into memory. CNN/FEMNIST models, GCN `X.npy`/`Y.npy`, tabular data, and other FedHPO algorithms are not used. The [FedHPO-Bench repository](https://github.com/FedHPO-Bench/FedHPO-Bench-ICLR23) documents the source archive. Load pickle files only from a trusted source.

The UCI-HAR runner also accepts `--download-data` when its configured dataset directory is empty. That option downloads the official data **and starts the experiment**.

## Configuration and outputs

| Experiment | Where to change settings |
| --- | --- |
| Synthetic GUIDE-FBO | `GUIDE-FBO/GUIDE_Config.py`: `FUNCTION_NAME`, `NUM_EXPERIMENTS`, `SEED_BASE`, `N_AGENTS`, `MAX_ITERATIONS`, `NOISE_SE`, `ACQ_FUNCTION`, `HETER_SHIFT`, and `HETER_ROTATION`. `M`, `SERVER_DISTRIBUTION_NUM`, `LAMBDA_MAX`, `RMS_EUCLIDEAN_THRESHOLD`, and `DPGMM_COVARIANCE_TYPE` control guidance. |
| Synthetic baselines | Top-level constants in each `Baselines/*.py` script, including its own `FUNCTION_NAME` and method-specific parameters. |
| Sensitivity and ablation | Top-level run switches, objective lists, and parameter value lists in `GUIDE-FBO/GUIDE_Sensitivity_Ablation.py`. |
| FedHPO | `CONFIG`, `GUIDE_CONFIG`, and `ALGORITHM_PARAMETERS` in `FedHPO_MultiTask_Runner.py`. |
| UCI-HAR | `CONFIG` and `UCI_HAR_GUIDE_CONFIG` in `UCI_HAR_MultiTask_Runner.py`; `--data-dir` overrides its data location. |
| Landmine | `CONFIG` and `GUIDE_CONFIG` in `Landmine_MultiTask_Runner.py`. |

Synthetic scripts write `*_btv.csv` (best-so-far objective), `*_instantaneous.csv` (current objective), and `*_timing.txt` in the working directory. Sensitivity and ablation outputs go under `results/`. Real-world runners write to result directories beside their scripts; Landmine additionally writes `*_test_auc.csv`. Metric CSVs have one iteration column and one column per random repetition. Timing depends on the machine.

This source release does not contain precomputed result CSVs. Run the corresponding script with the paper's settings to generate them. If paper-result CSVs are released later, they should be placed under `results/paper/` with a manifest that maps each file to its figure or table, algorithm, configuration, seeds, and dataset version. Preserve the dataset version, surrogate pickle set and order, seed settings, software environment, and hardware when comparing numerical results.

## Citation

Paper title: *[GUIDE-FBO: Guidance via Uncertainty Intervention and Distributional Exchange for Federated Bayesian Optimization](https://arxiv.org/abs/2609.35038)*.
