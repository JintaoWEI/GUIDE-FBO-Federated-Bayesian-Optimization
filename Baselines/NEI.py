import torch
import numpy as np
import pandas as pd
import random
import warnings
import time
import math
from torch import Tensor
from copy import deepcopy

from botorch.test_functions import (
    Ackley, Levy, Rosenbrock,
    Griewank, Rastrigin, StyblinskiTang, 
    Michalewicz, Powell
)
from botorch.models.gp_regression import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import LogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning


DEVICE = torch.device("cpu")
DTYPE = torch.double

FUNCTION_NAME = 'ellipsoid'  # Target Function Selection
# 'ackley', 'levy', 'griewank', 'rastrigin',
# 'rosenbrock', 'sphere', 'weierstrass', 'ellipsoid', 'zakharov', 
# 'michalewicz', 'powell', 'styblinskitang'

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

CURRENT_CONFIG = FUNCTION_CONFIGS[FUNCTION_NAME]
THEORETICAL_MAX = CURRENT_CONFIG.get('theoretical_max', 0.0)

DIM = CURRENT_CONFIG['dim']
NORMALIZE_X = CURRENT_CONFIG['normalize_x']
NORMALIZE_Y = CURRENT_CONFIG['normalize_y']
NOISE_SE = 0.1
SMALL_NUG = 1e-4

NUM_EXPERIMENTS = 10
SEED_BASE = 0
N_AGENTS = 16
INITIAL_SAMPLES = CURRENT_CONFIG['initial_samples']
MAX_ITERATIONS = 50

ENABLE_DEBUG_PRINT = True

# Heterogeneity levels (shift, rotation): homogeneous (0, 0), mild (0.05, 0.1), and severe (0.3, 1.0).
HETER_SHIFT = 0.3
HETER_ROTATION = 1.0

class Sphere:
    def __init__(self, dim=10, negate=False):
        self.dim = dim
        self.negate = negate
        self._optimal_value = 0.0
    
    def __call__(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        result = torch.sum(X**2, dim=-1)
        if self.negate:
            result = -result
        if result.numel() == 1:
            return result.squeeze()
        return result

class Weierstrass:
    def __init__(self, dim=10, negate=False):
        self.dim = dim
        self.negate = negate
        self._optimal_value = 0.0
        self.a = 0.5
        self.b = 3
        self.kmax = 21
    
    def __call__(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        device = X.device
        dtype = X.dtype
        obj = torch.zeros(X.shape[0], device=device, dtype=dtype)
        a_powers = torch.tensor([self.a**k for k in range(self.kmax)], device=device, dtype=dtype)
        b_powers = torch.tensor([self.b**k for k in range(self.kmax)], device=device, dtype=dtype)
        for i in range(X.shape[-1]):
            for k in range(self.kmax):
                obj += a_powers[k] * torch.cos(2 * torch.pi * b_powers[k] * (X[..., i] + 0.5))
        constant_term = torch.tensor(0.5, device=device, dtype=dtype)
        constant_sum = torch.zeros(1, device=device, dtype=dtype)
        for k in range(self.kmax):
            constant_sum += a_powers[k] * torch.cos(2 * torch.pi * b_powers[k] * constant_term)
        obj -= X.shape[-1] * constant_sum
        if self.negate:
            obj = -obj
        if obj.numel() == 1:
            return obj.squeeze()
        return obj

class Ellipsoid:
    def __init__(self, dim=10, negate=False):
        self.dim = dim
        self.negate = negate
        self._optimal_value = 0.0
    
    def __call__(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        weights = torch.arange(1, X.shape[-1] + 1, device=X.device, dtype=X.dtype)
        result = torch.sum(weights * X**2, dim=-1)
        if self.negate:
            result = -result
        if result.numel() == 1:
            return result.squeeze()
        return result

class Zakharov:
    def __init__(self, dim=10, negate=False):
        self.dim = dim
        self.negate = negate
        self._optimal_value = 0.0
    
    def __call__(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        term1 = torch.sum(X**2, dim=-1)
        indices = torch.arange(1, X.shape[-1] + 1, device=X.device, dtype=X.dtype)
        coeffs = 0.5 * indices
        weighted_sum = torch.sum(coeffs * X, dim=-1)
        term2 = weighted_sum ** 2
        term3 = weighted_sum ** 4
        result = term1 + term2 + term3
        if self.negate:
            result = -result
        if result.numel() == 1:
            return result.squeeze()
        return result

def get_blackbox_function(function_name, config):
    if function_name == 'ackley':
        return Ackley(dim=config['dim'], negate=config['negate'])
    elif function_name == 'levy':
        return Levy(negate=config['negate'], dim=config['dim'])
    elif function_name == 'griewank':
        return Griewank(dim=config['dim'], negate=config['negate'])
    elif function_name == 'rastrigin':
        return Rastrigin(dim=config['dim'], negate=config['negate'])
    elif function_name == 'sphere':
        return Sphere(dim=config['dim'], negate=config['negate'])
    elif function_name == 'rosenbrock':
        return Rosenbrock(dim=config['dim'], negate=config['negate'])
    elif function_name == 'weierstrass':
        return Weierstrass(dim=config['dim'], negate=config['negate'])
    elif function_name == 'ellipsoid':
        return Ellipsoid(dim=config['dim'], negate=config['negate'])
    elif function_name == 'zakharov':
        return Zakharov(dim=config['dim'], negate=config['negate'])
    elif function_name == 'michalewicz':
        return Michalewicz(dim=config['dim'], negate=config['negate'])
    elif function_name == 'powell':
        return Powell(dim=config['dim'], negate=config['negate'])
    elif function_name == 'styblinskitang':
        return StyblinskiTang(dim=config['dim'], negate=config['negate'])
    else:
        raise ValueError(f"Unsupported function: {function_name}")

blackbox = get_blackbox_function(FUNCTION_NAME, CURRENT_CONFIG)
bounds = torch.tensor(CURRENT_CONFIG['bounds'], device=DEVICE, dtype=DTYPE) / NORMALIZE_X

def truth(X):
    return blackbox(X * NORMALIZE_X) / NORMALIZE_Y


def equal_CLHS(n_agents, initial_samples, exp_id=0):
    # RNG state isolation begins.
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(SEED_BASE + exp_id * 10000)
    # ================================

    agents_cubes = []
    for agent_id in range(n_agents):
        agents_cubes.append([None, None])
    
    for dim in range(bounds.shape[1]):
        lb, ub = bounds[:, dim]
        divide_cube = torch.linspace(lb, ub, n_agents * initial_samples + 1)
        
        for i, agent_id in enumerate(torch.randperm(n_agents)):
            LHS_perm = torch.randperm(initial_samples)
            cube_length_value = (ub - lb) / (n_agents * initial_samples)
            
            if agents_cubes[agent_id][0] is None:
                agents_cubes[agent_id][0] = divide_cube[i:-1:n_agents][LHS_perm].unsqueeze(-1)
                agents_cubes[agent_id][1] = Tensor([cube_length_value])
            else:
                agents_cubes[agent_id][0] = torch.hstack([
                    agents_cubes[agent_id][0], 
                    divide_cube[i:-1:n_agents][LHS_perm].unsqueeze(-1)
                ])
                agents_cubes[agent_id][1] = torch.hstack([
                    agents_cubes[agent_id][1], 
                    Tensor([cube_length_value])
                ])
                
    # RNG state isolation ends.
    torch.set_rng_state(current_rng_state)
    # ================================
    
    return [(cube_lb, cube_length) for cube_lb, cube_length in agents_cubes]


# Local optimization agent for NEI.
class NEIAgent:
    def __init__(self, agent_id, cube_lb, cube_length, exp_id=0):
        self.agent_id = agent_id
        self.cube_lb = cube_lb.to(DEVICE)
        self.cube_length = cube_length.to(DEVICE)
        
        D = bounds.shape[1]
        
        # RNG state isolation begins.
        current_rng_state = torch.get_rng_state()
        unique_heter_seed = SEED_BASE + exp_id * 1000 + agent_id
        torch.manual_seed(unique_heter_seed)
        # ================================

        # 1. Generate the location shift vector z.
        domain_diagonal = torch.linalg.norm(bounds[1] - bounds[0])
        self.shift_z = torch.randn(1, D, device=DEVICE, dtype=DTYPE) * (HETER_SHIFT * domain_diagonal / math.sqrt(D))

        # 2. Generate the orthogonal rotation matrix R.
        if HETER_ROTATION > 0:
            random_M = torch.randn(D, D, device=DEVICE, dtype=DTYPE)
            skew_symmetric = random_M - random_M.T
            self.rotation_R = torch.matrix_exp(HETER_ROTATION * skew_symmetric)
        else:
            self.rotation_R = torch.eye(D, device=DEVICE, dtype=DTYPE)

        # RNG state isolation ends.
        torch.set_rng_state(current_rng_state)
        # ================================
        
        self.local_x = None
        self.local_y = None
        self.model = None
        self.mll = None
        self.optimized_param = None
        self.next_sample = None
        
        self.instantaneous_value = 0.0
        self.btv_value = -float('inf')
    
    def observation(self, X):
        # 1. Apply the input transform: shift by z, then rotate with R.T.
        shifted_X = X - self.shift_z
        transformed_X = torch.matmul(shifted_X, self.rotation_R.T)
        
        # 2. Evaluate the objective function.
        exact_y = truth(transformed_X) 
        
        if exact_y.dim() == 0: 
            exact_y = exact_y.unsqueeze(0).unsqueeze(1)
        elif exact_y.dim() == 1: 
            exact_y = exact_y.unsqueeze(-1)
        
        # 3. Update the observed value history.
        val = (exact_y[-1] * NORMALIZE_Y).item()
        self.instantaneous_value = val
        self.btv_value = max(self.btv_value, val)
        
        return exact_y + NOISE_SE * torch.randn_like(exact_y)
    
    def generate_initial_data(self, exp_id=0):
        # RNG state isolation begins.
        current_rng_state = torch.get_rng_state()
        init_data_seed = SEED_BASE + exp_id * 1000 + self.agent_id + 5000 
        torch.manual_seed(init_data_seed)
        # ================================

        self.local_x = torch.rand(INITIAL_SAMPLES, bounds.shape[1], device=DEVICE, dtype=DTYPE) \
                      * self.cube_length + self.cube_lb
        
        self.local_y = self.observation(self.local_x)
        
        with torch.no_grad():
            shifted_X = self.local_x - self.shift_z
            transformed_X = torch.matmul(shifted_X, self.rotation_R.T)
            all_true_y = truth(transformed_X) * NORMALIZE_Y
            self.btv_value = all_true_y.max().item()
        
        self.update_model()
        
        # RNG state isolation ends.
        torch.set_rng_state(current_rng_state)
        # ================================
    
    def update_model(self):
        noise_var = (NOISE_SE ** 2) * torch.ones_like(self.local_y)
        self.model = FixedNoiseGP(self.local_x, self.local_y, noise_var).to(DEVICE)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(DEVICE)
        fit_gpytorch_model(self.mll)
        self.optimized_param = deepcopy(self.model.state_dict())
    
    def get_next_design(self):
        num_restarts = 20
        raw_samples = 100
        batch_limit = 50
        max_itr = 25
        
        acq_func = LogNoisyExpectedImprovement(
            model=self.model,
            X_observed=self.local_x
        )
        
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": batch_limit, "maxiter": max_itr},
        )
        
        self.next_sample = candidates.detach()
    
    def get_next_observation(self):
        new_x = self.next_sample
        new_y = self.observation(new_x)
        self.local_x = torch.cat([self.local_x, new_x])
        self.local_y = torch.cat([self.local_y, new_y])
        self.update_model()
    
    def get_instantaneous_value(self):
        return self.instantaneous_value
    
    def get_btv(self):
        return self.btv_value


# Coordinates agent messages for NEI.
class NEIServer:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agents = []
    
    def initialize_agents(self, cubes, exp_id=0):
        for i in range(self.n_agents):
            # Pass the experiment ID to isolate heterogeneity sampling.
            agent = NEIAgent(i, cubes[i][0], cubes[i][1], exp_id) # (Keep the original agent class name.)
            # Pass the experiment ID to isolate initial sampling.
            agent.generate_initial_data(exp_id)
            self.agents.append(agent)
    
    def researching(self):
        for agent in self.agents:
            agent.get_next_design()
            agent.get_next_observation()


# Run one seeded experimental repetition.
def run_single_nei_experiment(exp_id):
    print(f"\n=== Running Experiment {exp_id + 1}/{NUM_EXPERIMENTS} ===")
    exp_start_time = time.time()
    
    seed = SEED_BASE + exp_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    cubes = equal_CLHS(N_AGENTS, INITIAL_SAMPLES, exp_id)
    
    server = NEIServer(N_AGENTS)
    # 2. Pass the experiment ID to the server scheduler.
    server.initialize_agents(cubes, exp_id)
    
    avg_btv_values = []
    avg_instantaneous_values = []
    
    print(f"Iter 0 | Initialization Complete (Init Strategy: CLHS)")
    
    btv_values = [agent.get_btv() for agent in server.agents]
    instantaneous_values = [agent.get_instantaneous_value() for agent in server.agents]
    
    avg_btv = np.mean(btv_values)
    avg_instantaneous = np.mean(instantaneous_values)
    
    avg_btv_values.append(avg_btv)
    avg_instantaneous_values.append(avg_instantaneous)
    
    print(f"Iter 0 | Avg BTV: {avg_btv:.4f} | Avg Instantaneous: {avg_instantaneous:.4f}")
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        if ENABLE_DEBUG_PRINT:
            print(f"\n--- Iter {iteration} ---")
        
        server.researching()
        
        btv_values = [agent.get_btv() for agent in server.agents]
        instantaneous_values = [agent.get_instantaneous_value() for agent in server.agents]
        
        avg_btv = np.mean(btv_values)
        avg_instantaneous = np.mean(instantaneous_values)
        
        avg_btv_values.append(avg_btv)
        avg_instantaneous_values.append(avg_instantaneous)
        
        if ENABLE_DEBUG_PRINT:
            print(f"Iter {iteration} | Avg BTV: {avg_btv:.4f} | Avg Instantaneous: {avg_instantaneous:.4f}")
    
    exp_elapsed_time = time.time() - exp_start_time
    
    return avg_btv_values, avg_instantaneous_values, exp_elapsed_time
def main():
    print(f"=== NEI Algorithm Experiment ===")
    print(f"Function: {FUNCTION_NAME.upper()}")
    print(f"Dim: {DIM}")
    print(f"Theoretical Max: {THEORETICAL_MAX}")
    print(f"Algorithm Config:")
    print(f"  - Init Strategy: CLHS")
    print(f"  - Acquisition: LogNoisyExpectedImprovement")
    print(f"  - Collaboration: None (Independent Optimization)")
    print(f"Experiment Params: N_agents={N_AGENTS}, MAX_ITERATIONS={MAX_ITERATIONS}")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    
    all_btv_results = []
    all_instantaneous_results = []
    all_timings = []
    
    for exp_id in range(NUM_EXPERIMENTS):
        btv_vals, instantaneous_vals, elapsed_time = run_single_nei_experiment(exp_id)
        all_btv_results.append(btv_vals)
        all_instantaneous_results.append(instantaneous_vals)
        all_timings.append(elapsed_time)
    
    output_prefix = f"NEI_{FUNCTION_NAME.upper()}{DIM}_S{HETER_SHIFT}_R{HETER_ROTATION}"
    iterations = list(range(MAX_ITERATIONS + 1))
    
    btv_df = pd.DataFrame({'iteration': iterations})
    for exp_id in range(NUM_EXPERIMENTS):
        btv_df[f'exp_{exp_id + 1}'] = all_btv_results[exp_id]
    btv_csv = f"{output_prefix}_btv.csv"
    btv_df.to_csv(btv_csv, index=False)
    print(f"\nBTV values saved to: {btv_csv}")
    
    instantaneous_df = pd.DataFrame({'iteration': iterations})
    for exp_id in range(NUM_EXPERIMENTS):
        instantaneous_df[f'exp_{exp_id + 1}'] = all_instantaneous_results[exp_id]
    instantaneous_csv = f"{output_prefix}_instantaneous.csv"
    instantaneous_df.to_csv(instantaneous_csv, index=False)
    print(f"Instantaneous values saved to: {instantaneous_csv}")
    
    final_avg_btv = np.mean([results[-1] for results in all_btv_results])
    final_avg_instantaneous = np.mean([results[-1] for results in all_instantaneous_results])
    
    print(f"\n=== Experiment Finished ===")
    print(f"Final Avg BTV Value: {final_avg_btv:.4f}")
    print(f"Final Avg Instantaneous Value: {final_avg_instantaneous:.4f}")
    
    print(f"\n=== Timing Stats ===")
    timing_lines = []
    for exp_id, elapsed_time in enumerate(all_timings):
        timing_line = f"Exp {exp_id + 1} Time: {elapsed_time:.2f} s"
        print(timing_line)
        timing_lines.append(timing_line)
    
    avg_time = np.mean(all_timings)
    total_time = np.sum(all_timings)
    
    print("---")
    avg_line = f"Avg Time: {avg_time:.2f} s"
    total_line = f"Total Time: {total_time:.2f} s"
    print(avg_line)
    print(total_line)
    
    timing_lines.append("---")
    timing_lines.append(avg_line)
    timing_lines.append(total_line)
    
    timing_txt = f"{output_prefix}_timing.txt"
    with open(timing_txt, 'w', encoding='utf-8') as f:
        f.write("=== NEI Algorithm Timing Stats ===\n")
        for line in timing_lines:
            f.write(line + "\n")
    
    print(f"\nTiming stats saved to: {timing_txt}")
    print(f"\n=== Config Summary ===")
    print(f"Function: {FUNCTION_NAME} | Dim: {DIM} | Init: CLHS")
    print(f"Agents: {N_AGENTS} | Collaboration: None (Independent Baseline)")

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=InputDataWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    main()