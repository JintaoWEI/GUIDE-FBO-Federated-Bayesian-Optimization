"""GUIDE sensitivity and ablation experiments driven by top-level constants."""

import copy
import importlib
import time
from pathlib import Path

import pandas as pd

import GUIDE_Config as config
import GUIDE_Utils
import GUIDE_Orf
import GUIDE_Agent
import GUIDE_Server
import GUIDE_Main


# ================== User controls ==================
RUN_SENSITIVITY = False
RUN_ABLATION = True
OVERWRITE_EXISTING = False

SENSITIVITY_FUNCTIONS = ['zakharov']
ABLATION_FUNCTIONS = ['michalewicz']
OUTPUT_ROOT = 'results'

M_VALUES = [100, 250, 500, 750, 1000]
P_VALUES = [1, 3, 5, 8, 10]
LAMBDA_MAX_VALUES = [0.25, 0.5, 1.0, 2.0]
RMS_EUCLIDEAN_THRESHOLD_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05]

_CONFIG_NAMES = [
    'FUNCTION_NAME', 'NUM_EXPERIMENTS', 'SEED_BASE', 'N_AGENTS',
    'INITIAL_SAMPLES', 'MAX_ITERATIONS', 'HETER_SHIFT', 'HETER_ROTATION',
    'M', 'ORF_NUM_SAMPLES',
    'SERVER_DISTRIBUTION_NUM', 'LAMBDA_MAX', 'RMS_EUCLIDEAN_THRESHOLD',
    'DPGMM_COVARIANCE_TYPE', 'ACQ_FUNCTION',
    'ENABLE_MERGING', 'ENABLE_WEIGHT_CALIBRATION',
    'FULL_GLOBAL_PACKET', 'DIRECT_SURROGATE_INTERVENTION',
    'BELIEF_EXTRACTION_MODE', 'AGENT_SPECIFIC_SAMPLING',
    'SPATIAL_GUIDANCE', 'GUIDANCE_AVERAGE_NUM_POINTS',
    'ACQ_NUM_RESTARTS', 'ACQ_RAW_SAMPLES', 'ACQ_BATCH_LIMIT',
    'ACQ_MAX_ITER',
]
_DEFAULT_CONFIG = {
    name: copy.deepcopy(getattr(config, name)) for name in _CONFIG_NAMES
}


def _restore_defaults(function_name):
    """Restore Config globals and recompute all function-derived constants."""
    for name, value in _DEFAULT_CONFIG.items():
        setattr(config, name, copy.deepcopy(value))
    config.FUNCTION_NAME = function_name
    config.CURRENT_CONFIG = config.FUNCTION_CONFIGS[function_name]
    config.THEORETICAL_MAX = config.CURRENT_CONFIG.get('theoretical_max', 0.0)
    config.DIM = config.CURRENT_CONFIG['dim']
    config.NORMALIZE_X = config.CURRENT_CONFIG['normalize_x']
    config.NORMALIZE_Y = config.CURRENT_CONFIG['normalize_y']
    config.INITIAL_SAMPLES = config.CURRENT_CONFIG['initial_samples']
    _refresh_output_names()


def _refresh_output_names():
    prefix = config.get_output_filename_prefix()
    config.BTV_CSV = f'{prefix}_btv.csv'
    config.INSTANTANEOUS_CSV = f'{prefix}_instantaneous.csv'
    config.TIMING_TXT = f'{prefix}_timing.txt'


def _reload_algorithm_modules():
    """Reload in dependency order after Config globals have been updated."""
    global GUIDE_Utils, GUIDE_Orf, GUIDE_Agent, GUIDE_Server, GUIDE_Main
    GUIDE_Utils = importlib.reload(GUIDE_Utils)
    GUIDE_Orf = importlib.reload(GUIDE_Orf)
    GUIDE_Agent = importlib.reload(GUIDE_Agent)
    GUIDE_Server = importlib.reload(GUIDE_Server)
    GUIDE_Main = importlib.reload(GUIDE_Main)


def _run_seed(function_name, variant, seed_index, overrides):
    """Run one seed and return its two histories and total elapsed time."""
    _restore_defaults(function_name)
    for name, value in overrides.items():
        setattr(config, name, value)
    _refresh_output_names()
    _reload_algorithm_modules()

    print(f'[run] {variant} | {function_name} | seed={seed_index}')
    btv, instantaneous, elapsed = GUIDE_Main.run_single_experiment(seed_index)
    if len(btv) != len(instantaneous):
        raise ValueError(
            f'BTV and instantaneous result lengths differ for seed {seed_index}: '
            f'{len(btv)} != {len(instantaneous)}'
        )
    return btv, instantaneous, elapsed


def _seed_count():
    return int(_DEFAULT_CONFIG['NUM_EXPERIMENTS'])


def _expected_iteration_count():
    return int(_DEFAULT_CONFIG['MAX_ITERATIONS']) + 1


def _metric_csv_is_complete(output_csv):
    """Check one metric CSV against the reference wide-table layout."""
    if not output_csv.is_file():
        return False

    try:
        result = pd.read_csv(output_csv)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False

    expected_columns = ['iteration'] + [
        f'exp_{seed_index + 1}' for seed_index in range(_seed_count())
    ]
    expected_iterations = list(range(_expected_iteration_count()))
    return (
        result.columns.tolist() == expected_columns
        and result['iteration'].tolist() == expected_iterations
        and not result.isna().any().any()
    )


def _result_is_complete(output_prefix):
    """Return whether all output files for one setting are complete."""
    btv_csv = output_prefix.parent / f'{output_prefix.name}_btv.csv'
    instantaneous_csv = (
        output_prefix.parent / f'{output_prefix.name}_instantaneous.csv'
    )
    timing_txt = output_prefix.parent / f'{output_prefix.name}_timing.txt'
    if not (
        _metric_csv_is_complete(btv_csv)
        and _metric_csv_is_complete(instantaneous_csv)
        and timing_txt.is_file()
    ):
        return False

    try:
        timing_text = timing_txt.read_text(encoding='utf-8')
    except OSError:
        return False
    return all(
        f'Exp {seed_index + 1} ' in timing_text
        for seed_index in range(_seed_count())
    )


def _build_metric_dataframe(seed_results, result_index):
    """Arrange one metric as iteration rows and random-seed columns."""
    expected_length = _expected_iteration_count()
    result = pd.DataFrame({'iteration': list(range(expected_length))})
    for seed_index, seed_result in enumerate(seed_results):
        values = seed_result[result_index]
        if len(values) != expected_length:
            raise ValueError(
                f'Unexpected iteration count for seed {seed_index}: '
                f'{len(values)} != {expected_length}'
            )
        result[f'exp_{seed_index + 1}'] = values
    return result


def _write_csv_atomically(dataframe, output_csv):
    """Write a CSV without exposing a partially written final file."""
    temporary_csv = output_csv.with_suffix('.tmp.csv')
    dataframe.to_csv(temporary_csv, index=False)
    temporary_csv.replace(output_csv)


def _write_timing_file(seed_results, output_txt):
    """Save one total elapsed time per random seed as a text report."""
    elapsed_times = [float(seed_result[2]) for seed_result in seed_results]
    timing_lines = ['=== Timing Stats ===']
    for seed_index, elapsed in enumerate(elapsed_times):
        seed = int(_DEFAULT_CONFIG['SEED_BASE']) + seed_index
        timing_lines.append(
            f'Exp {seed_index + 1} (seed={seed}) Time: {elapsed:.6f} s'
        )
    timing_lines.extend([
        '---',
        f'Avg Time: {sum(elapsed_times) / len(elapsed_times):.6f} s',
        f'Total Time: {sum(elapsed_times):.6f} s',
    ])

    temporary_txt = output_txt.with_suffix('.tmp.txt')
    temporary_txt.write_text('\n'.join(timing_lines) + '\n', encoding='utf-8')
    temporary_txt.replace(output_txt)


def _run_experiment_group(function_name, variant, overrides, output_prefix):
    """Run all seeds and save two wide CSV files plus one timing report."""
    if _result_is_complete(output_prefix) and not OVERWRITE_EXISTING:
        print(f'[skip] complete result: {output_prefix}')
        return

    seed_results = [
        _run_seed(function_name, variant, seed_index, overrides)
        for seed_index in range(_seed_count())
    ]
    btv_df = _build_metric_dataframe(seed_results, result_index=0)
    instantaneous_df = _build_metric_dataframe(seed_results, result_index=1)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    btv_csv = output_prefix.parent / f'{output_prefix.name}_btv.csv'
    instantaneous_csv = (
        output_prefix.parent / f'{output_prefix.name}_instantaneous.csv'
    )
    timing_txt = output_prefix.parent / f'{output_prefix.name}_timing.txt'
    _write_csv_atomically(btv_df, btv_csv)
    _write_csv_atomically(instantaneous_df, instantaneous_csv)
    _write_timing_file(seed_results, timing_txt)
    print(f'[saved] {btv_csv}')
    print(f'[saved] {instantaneous_csv}')
    print(f'[saved] {timing_txt}')


def _sensitivity_specs():
    return [
        ('M', 'M', M_VALUES),
        ('P', 'SERVER_DISTRIBUTION_NUM', P_VALUES),
        ('lambda_max', 'LAMBDA_MAX', LAMBDA_MAX_VALUES),
        (
            'rms_euclidean_distance_threshold',
            'RMS_EUCLIDEAN_THRESHOLD',
            RMS_EUCLIDEAN_THRESHOLD_VALUES,
        ),
    ]

def run_sensitivity():
    # Use the acquisition name to distinguish result directories and files.
    acq_name = str(config.ACQ_FUNCTION).upper()

    for function_name in SENSITIVITY_FUNCTIONS:
        for directory_name, config_name, values in _sensitivity_specs():
            for value in values:
                output_prefix = (
                    Path(OUTPUT_ROOT)
                    / 'sensitivity'
                    / function_name
                    / acq_name
                    / directory_name
                    / f'{acq_name}_{directory_name}_value_{value}'
                )
                _run_experiment_group(
                    function_name=function_name,
                    variant='full_guide_diag',
                    overrides={config_name: value},
                    output_prefix=output_prefix,
                )


_ABLATION_VARIANTS = {
    'full_guide_diag': {},
    'single_gaussian_belief': {'BELIEF_EXTRACTION_MODE': 'single_gaussian'},
    'no_value_aware_reweighting': {'ENABLE_WEIGHT_CALIBRATION': False},
    'no_component_merging': {'ENABLE_MERGING': False},
    'no_agent_specific_sampling': {'AGENT_SPECIFIC_SAMPLING': False},
    'uniform_uncertainty_scaling': {'SPATIAL_GUIDANCE': False},
}


def run_ablation():
    for function_name in ABLATION_FUNCTIONS:
        for variant, overrides in _ABLATION_VARIANTS.items():
            # Level 3: heavy heterogeneity.
            ablation_overrides = {
                'ACQ_FUNCTION': 'UCB',
                'DPGMM_COVARIANCE_TYPE': 'diag',
                'HETER_SHIFT': 0.3,
                'HETER_ROTATION': 1.0,
                **overrides,
            }
            output_prefix = (
                Path(OUTPUT_ROOT) / 'ablation_level3' / function_name
                / variant / variant
            )
            _run_experiment_group(
                function_name=function_name,
                variant=variant,
                overrides=ablation_overrides,
                output_prefix=output_prefix,
            )


def main():
    start = time.time()
    if RUN_SENSITIVITY:
        run_sensitivity()
    if RUN_ABLATION:
        run_ablation()
    _restore_defaults(_DEFAULT_CONFIG['FUNCTION_NAME'])
    _reload_algorithm_modules()
    print(f'GUIDE sensitivity/ablation finished in {time.time() - start:.2f} s')


if __name__ == '__main__':
    main()
