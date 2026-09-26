"""
GUIDE-FBO configuration
Defines experiment settings, benchmark functions, and algorithm hyperparameters.
"""
import os
import torch
# ================== Runtime settings ==================
DEVICE = torch.device("cpu")
DTYPE = torch.double

# ================== Benchmark function selection ==================
# Supported benchmark functions (12):
# 'ackley', 'levy', 'griewank', 'rastrigin',
# 'rosenbrock', 'sphere', 'weierstrass', 'ellipsoid', 'zakharov', 
# 'michalewicz', 'powell', 'styblinskitang'
FUNCTION_NAME = 'ackley'
# ================== Benchmark function selection ==================
# Optional GUIDE_FUNC environment variable example; the active setting is FUNCTION_NAME.
# FUNCTION_NAME = os.environ.get('GUIDE_FUNC', 'ackley')

# Benchmark function settings
FUNCTION_CONFIGS = {
    'ackley': {
        'dim': 10,
        'normalize_x': 32.768 * 2,
        'normalize_y': 20,
        'bounds': [[-32.768] * 10, [32.768] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'levy': {
        'dim': 10,
        'normalize_x': 20,
        'normalize_y': 200,
        'bounds': [[-10] * 10, [10] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'griewank': {
        'dim': 10,
        'normalize_x': 1200,
        'normalize_y': 200,
        'bounds': [[-600] * 10, [600] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'rastrigin': {
        'dim': 10,
        'normalize_x': 10.24,
        'normalize_y': 200,
        'bounds': [[-5.12] * 10, [5.12] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'rosenbrock': {
        'dim': 10,
        'normalize_x': 4.096,
        'normalize_y': 5000,
        'bounds': [[-2.048] * 10, [2.048] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'sphere': {
        'dim': 10,
        'normalize_x': 10,
        'normalize_y': 200,
        'bounds': [[-5] * 10, [5] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'weierstrass': {
        'dim': 10,
        'normalize_x': 1,
        'normalize_y': 50,
        'bounds': [[-0.5] * 10, [0.5] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'ellipsoid': {
        'dim': 10,
        'normalize_x': 10.24,
        'normalize_y': 2000,
        'bounds': [[-5.12] * 10, [5.12] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'zakharov': {
        'dim': 10,
        'normalize_x': 10,
        'normalize_y': 1e7,
        'bounds': [[-5] * 10, [5] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'michalewicz': {
        'dim': 10,
        'normalize_x': 3.14159265359,
        'normalize_y': 10,
        'bounds': [[0] * 10, [3.14159265359] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 9.66
    },
    'powell': {
        'dim': 10,
        'normalize_x': 10,
        'normalize_y': 5000,
        'bounds': [[-5] * 10, [5] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 0.0
    },
    'styblinskitang': {
        'dim': 10,
        'normalize_x': 10,
        'normalize_y': 400,
        'bounds': [[-5] * 10, [5] * 10],
        'initial_samples': 30,
        'negate': True,
        'theoretical_max': 391.66
    }
}

# Select the active benchmark settings.
CURRENT_CONFIG = FUNCTION_CONFIGS[FUNCTION_NAME]
THEORETICAL_MAX = CURRENT_CONFIG.get('theoretical_max', 0.0)

# ================== Experiment and environment settings ==================
NUM_EXPERIMENTS = 10        # Number of independent repetitions
SEED_BASE = 0               # Base random seed
DIM = CURRENT_CONFIG['dim'] # Objective dimension
NORMALIZE_X = CURRENT_CONFIG['normalize_x']
NORMALIZE_Y = CURRENT_CONFIG['normalize_y']
NOISE_SE = 0.1              # Observation noise standard deviation
N_AGENTS = 16               # Number of federated agents
INITIAL_SAMPLES = CURRENT_CONFIG['initial_samples'] # Initial samples per agent
MAX_ITERATIONS = 50         # Bayesian optimization rounds
# ================== Federated heterogeneity settings (shift z and rotation R) ==================
# Homogeneous: HETER_SHIFT = 0.0, HETER_ROTATION = 0.0
# Mild heterogeneity: HETER_SHIFT = 0.05, HETER_ROTATION = 0.1 (about 6.5 degrees)
# Severe heterogeneity: HETER_SHIFT = 0.3, HETER_ROTATION = 1.0 (about 65 degrees)

# Shift magnitude as a fraction of the input domain width.
HETER_SHIFT = 0.05
# Rotation magnitude controlling variable coupling.
# An antisymmetric matrix exponential gives an orthogonal transform; larger values rotate more.
HETER_ROTATION = 0.1
# ================== Algorithm parameters ==================
# Thompson sampling and ORF settings
M = 500                     # Number of Thompson samples used to extract promising regions
USE_ORF_FOR_TS = True       # Use orthogonal random features for extraction-stage Thompson sampling
ORF_NUM_SAMPLES = 500       # Number of orthogonal random features

DPGMM_COVARIANCE_TYPE = 'diag'    # DPGMM covariance structure: 'full' or 'diag'

# Federated augmented acquisition settings
ACQ_FUNCTION = 'UCB'        # Base acquisition function: 'UCB', 'NEI', or 'TS'
ACQ_NUM_RESTARTS = 20
ACQ_RAW_SAMPLES = 100
ACQ_BATCH_LIMIT = 50
ACQ_MAX_ITER = 25

# === Guidance strength settings ===
LAMBDA_MAX = 1.0               # Maximum global guidance strength lambda
RMS_EUCLIDEAN_THRESHOLD = 0.05  # Server clustering threshold: RMS Euclidean distance between component means

SERVER_DISTRIBUTION_NUM = 5    # Number of server components sent to each agent per round (P)

# ================== Ablation switches ==================
# Defaults implement full GUIDE with diagonal covariance. The sensitivity runner changes these globals
# and then reloads Utils, ORF, Agent, Server, and Main in dependency order.
ENABLE_MERGING = True
ENABLE_WEIGHT_CALIBRATION = True
FULL_GLOBAL_PACKET = False
DIRECT_SURROGATE_INTERVENTION = False

# Single-factor ablation switches; defaults preserve full GUIDE-FBO behavior.
BELIEF_EXTRACTION_MODE = 'dpgmm'  # Supported: 'dpgmm' or 'single_gaussian'
AGENT_SPECIFIC_SAMPLING = True
SPATIAL_GUIDANCE = True
GUIDANCE_AVERAGE_NUM_POINTS = 512

FIXED_BETA = 2.0               # UCB exploration parameter beta

VISUALIZE_PROCESS = False      # Print the five highest calibrated server weights when enabled.

# ================== Output file names ==================
def get_output_filename_prefix():
    return f"GUIDE_{FUNCTION_NAME.upper()}{DIM}_{ACQ_FUNCTION}_lambda{LAMBDA_MAX}_{DPGMM_COVARIANCE_TYPE}_S{HETER_SHIFT}_R{HETER_ROTATION}"

BTV_CSV = f"{get_output_filename_prefix()}_btv.csv" # Best-so-far true objective value
INSTANTANEOUS_CSV = f"{get_output_filename_prefix()}_instantaneous.csv" # Instantaneous observed value
TIMING_TXT = f"{get_output_filename_prefix()}_timing.txt" # Elapsed-time report
