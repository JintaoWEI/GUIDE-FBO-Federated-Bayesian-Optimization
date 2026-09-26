import torch
import numpy as np
import pandas as pd
import random
import warnings
import time
import math
from torch import Tensor
from copy import deepcopy
from contextlib import ExitStack

from botorch.test_functions import (
    Ackley, Levy, Rosenbrock,
    Griewank, Rastrigin, StyblinskiTang, 
    Michalewicz, Powell
)
from botorch.models.gp_regression import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import AnalyticAcquisitionFunction, ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions import Normal


DEVICE = torch.device("cpu")
DTYPE = torch.double

# Function Selection
# 'ackley', 'levy', 'griewank', 'rastrigin',
# 'rosenbrock', 'sphere', 'weierstrass', 'ellipsoid', 'zakharov', 
# 'michalewicz', 'powell', 'styblinskitang'
FUNCTION_NAME = 'ellipsoid'  # Change this to select different functions

# Function Configuration Dictionary
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
    'sphere': {
        'dim': 10,
        'normalize_x': 10,
        'normalize_y': 200,
        'bounds': [[-5] * 10, [5] * 10],
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

# Get current function config
CURRENT_CONFIG = FUNCTION_CONFIGS[FUNCTION_NAME]
THEORETICAL_MAX = CURRENT_CONFIG.get('theoretical_max', 0.0)

# Dynamic objective function settings
DIM = CURRENT_CONFIG['dim']
NORMALIZE_X = CURRENT_CONFIG['normalize_x']
NORMALIZE_Y = CURRENT_CONFIG['normalize_y']
NOISE_SE = 0.1
SMALL_NUG = 1e-4

# Experiment Settings
NUM_EXPERIMENTS = 10
SEED_BASE = 0
N_AGENTS = 16
INITIAL_SAMPLES = CURRENT_CONFIG['initial_samples']
MAX_ITERATIONS = 50
# Heterogeneity levels (shift, rotation): homogeneous (0, 0), mild (0.05, 0.1), and severe (0.3, 1.0).
HETER_SHIFT = 0.3
HETER_ROTATION = 1.0

# FMTBO Specific Parameters
FMTBO_KT_AGG_PROB = 0.8  # Probability of aggregation in Knowledge Transfer (kt_agg_prob)
FMTBO_GAMMA = 0.5        # Trade-off parameter for Weighted EI
FMTBO_ENABLE_KT = True   # Enable Knowledge Transfer (knowledge_transfer)
FMTBO_ACQ_TYPE = 'EI_w'  # 'EI' or 'EI_w'

# Debug switch
ENABLE_DEBUG_PRINT = True

#Custom Test Functions

class Sphere:
    """Sphere function"""
    def __init__(self, dim=10, negate=False):
        self.dim = dim
        self.negate = negate
    
    def __call__(self, X):
        if X.dim() == 1: X = X.unsqueeze(0)
        result = torch.sum(X**2, dim=-1)
        if self.negate: result = -result
        return result.squeeze() if result.numel() == 1 else result

class Weierstrass:
    """Weierstrass function"""
    def __init__(self, dim=10, negate=False):
        self.dim = dim
        self.negate = negate
        self.a = 0.5
        self.b = 3
        self.kmax = 21
    
    def __call__(self, X):
        if X.dim() == 1: X = X.unsqueeze(0)
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
        if self.negate: obj = -obj
        return obj.squeeze() if obj.numel() == 1 else obj

class Ellipsoid:
    """Ellipsoid function"""
    def __init__(self, dim=10, negate=False):
        self.dim = dim
        self.negate = negate
    
    def __call__(self, X):
        if X.dim() == 1: X = X.unsqueeze(0)
        weights = torch.arange(1, X.shape[-1] + 1, device=X.device, dtype=X.dtype)
        result = torch.sum(weights * X**2, dim=-1)
        if self.negate: result = -result
        return result.squeeze() if result.numel() == 1 else result

class Zakharov:
    """Zakharov function"""
    def __init__(self, dim=10, negate=False):
        self.dim = dim
        self.negate = negate
    
    def __call__(self, X):
        if X.dim() == 1: X = X.unsqueeze(0)
        term1 = torch.sum(X**2, dim=-1)
        indices = torch.arange(1, X.shape[-1] + 1, device=X.device, dtype=X.dtype)
        coeffs = 0.5 * indices
        weighted_sum = torch.sum(coeffs * X, dim=-1)
        term2 = weighted_sum ** 2
        term3 = weighted_sum ** 4
        result = term1 + term2 + term3
        if self.negate: result = -result
        return result.squeeze() if result.numel() == 1 else result

# Get blackbox function
def get_blackbox_function(function_name, config):
    """Get the corresponding blackbox function by name"""
    if function_name == 'ackley': return Ackley(dim=config['dim'], negate=config['negate'])
    elif function_name == 'levy': return Levy(negate=config['negate'], dim=config['dim'])
    elif function_name == 'griewank': return Griewank(dim=config['dim'], negate=config['negate'])
    elif function_name == 'rastrigin': return Rastrigin(dim=config['dim'], negate=config['negate'])
    elif function_name == 'sphere': return Sphere(dim=config['dim'], negate=config['negate'])
    elif function_name == 'rosenbrock': return Rosenbrock(dim=config['dim'], negate=config['negate'])
    elif function_name == 'weierstrass': return Weierstrass(dim=config['dim'], negate=config['negate'])
    elif function_name == 'ellipsoid': return Ellipsoid(dim=config['dim'], negate=config['negate'])
    elif function_name == 'zakharov': return Zakharov(dim=config['dim'], negate=config['negate'])
    elif function_name == 'michalewicz': return Michalewicz(dim=config['dim'], negate=config['negate'])
    elif function_name == 'powell': return Powell(dim=config['dim'], negate=config['negate'])
    elif function_name == 'styblinskitang': return StyblinskiTang(dim=config['dim'], negate=config['negate'])
    else: raise ValueError(f"Unsupported function: {function_name}")

# Initialize blackbox and bounds
blackbox = get_blackbox_function(FUNCTION_NAME, CURRENT_CONFIG)
bounds = torch.tensor(CURRENT_CONFIG['bounds'], device=DEVICE, dtype=DTYPE) / NORMALIZE_X

def truth(X):
    """True function evaluation, applying normalization"""
    return blackbox(X * NORMALIZE_X) / NORMALIZE_Y


def equal_CLHS(n_agents, initial_samples, exp_id=0):
    """
    Collaborative Latin Hypercube Sampling for equal initialization
    """
    # RNG state isolation begins.
    current_rng_state = torch.get_rng_state()
    # Use a dedicated seed for common Latin hypercube sampling.
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


class WeightedEI(AnalyticAcquisitionFunction):
    """
    Weighted Expected Improvement (EI_w) for FMTBO
    Combines Global EI and Local EI:
    EI_w(x) = gamma * EI_global(x) + (1 - gamma) * EI_local(x)
    Reference: deploy/clientgp.py -> EI_w
    """
    def __init__(self, global_model, local_model, best_f, gamma, **kwargs):
        super().__init__(model=global_model, **kwargs)
        self.local_model = local_model
        self.gamma = gamma
        self.best_f = best_f
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean_g, sigma_g = self._mean_and_sigma(X, self.model)
        ei_g = self._compute_ei(mean_g, sigma_g)
        
        if self.local_model is not None:
            mean_l, sigma_l = self._mean_and_sigma(X, self.local_model)
            ei_l = self._compute_ei(mean_l, sigma_l)
            return self.gamma * ei_g + (1 - self.gamma) * ei_l
        else:
            return ei_g

    def _mean_and_sigma(self, X, model):
        posterior = model.posterior(X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        sigma = posterior.variance.sqrt().squeeze(-2).squeeze(-1)
        return mean, sigma

    def _compute_ei(self, mean, sigma):
        u = (mean - self.best_f) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (u * ucdf + updf)
        return ei


class FMTBOAgent:
    """Agent class for FMTBO Algorithm"""
    
    def __init__(self, agent_id, cube_lb, cube_length, exp_id=0):
        self.agent_id = agent_id
        self.cube_lb = cube_lb.to(DEVICE)
        self.cube_length = cube_length.to(DEVICE)
        
        # Heterogeneity
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
        
        # Data storage
        self.local_x = None
        self.local_y = None
        
        # Models
        self.global_gp_model = None
        self.local_gp_model = None
        
        self.btv_value = -float('inf')
        self.instantaneous_value = 0.0
        
        # Current shared-set rank for KT
        self.current_rank = None

    def observation(self, X):
        """Observe function value (with noise) and store true value"""
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
        """Generate initial data using CLHS"""
        
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
            
        # Initial fit with no global knowledge
        self.fit_models(global_state_dict=None)

        # RNG state isolation ends.
        torch.set_rng_state(current_rng_state)
        # ================================

    def fit_models(self, global_state_dict=None):

        noise_var = (NOISE_SE ** 2) * torch.ones_like(self.local_y)
        
        # 1. Fit Global Model
        self.global_gp_model = FixedNoiseGP(self.local_x, self.local_y, noise_var).to(DEVICE)
        
        if global_state_dict is not None:
            # Load federated parameters as initialization
            self.global_gp_model.load_state_dict(global_state_dict)
            
        # Optimization (Fine-tuning on local data)
        mll_g = ExactMarginalLogLikelihood(self.global_gp_model.likelihood, self.global_gp_model).to(DEVICE)
        fit_gpytorch_model(mll_g)
        
        # 2. Fit Local Model (Fresh initialization)
        if FMTBO_ACQ_TYPE == 'EI_w':
            self.local_gp_model = FixedNoiseGP(self.local_x, self.local_y, noise_var).to(DEVICE)
            mll_l = ExactMarginalLogLikelihood(self.local_gp_model.likelihood, self.local_gp_model).to(DEVICE)
            fit_gpytorch_model(mll_l)

    def get_model_params(self):
        return deepcopy(self.global_gp_model.state_dict())

    def calculate_rank_on_shared(self, shared_x):
        if self.global_gp_model is None:
            return None
        with torch.no_grad():
            posterior = self.global_gp_model.posterior(shared_x)
            mean_preds = posterior.mean.squeeze()
            
            sorted_idx = torch.argsort(mean_preds)
            
            ranks = torch.argsort(sorted_idx)
            
            return ranks.cpu().numpy()

    def select_next_point(self):
        """Optimize acquisition function to select next point"""
        best_f = self.local_y.max().item()
        
        # Define Acquisition Function
        if FMTBO_ACQ_TYPE == 'EI_w':
            acq = WeightedEI(
                global_model=self.global_gp_model,
                local_model=self.local_gp_model,
                best_f=best_f,
                gamma=FMTBO_GAMMA
            )
        else:
            acq = ExpectedImprovement(model=self.global_gp_model, best_f=best_f)
            
        num_restarts = 20
        raw_samples = 100
        batch_limit = 50
        max_itr = 25
        
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": batch_limit, "maxiter": max_itr}
        )
        
        new_x = candidates.detach()
        new_y = self.observation(new_x)
        
        self.local_x = torch.cat([self.local_x, new_x])
        self.local_y = torch.cat([self.local_y, new_y])
        
        return new_x

    def get_btv_value(self): return self.btv_value
    def get_instantaneous_value(self): return self.instantaneous_value



class FMTBOServer:
    """Server class for FMTBO (Federated Averaging + Knowledge Transfer)"""
    
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agents = []
        self.global_model_params = None
        self.shared_x = None # Shared validation set for KT
        
    def initialize_agents(self, cubes, exp_id=0):
        """Initialize all agents"""
        for i in range(self.n_agents):
            # Pass the experiment ID to isolate heterogeneity generation.
            agent = FMTBOAgent(i, cubes[i][0], cubes[i][1], exp_id)
            # Pass the experiment ID to isolate initial sampling.
            agent.generate_initial_data(exp_id)
            self.agents.append(agent)

        self.global_model_params = [deepcopy(self.agents[0].get_model_params()) for _ in range(self.n_agents)]
        
        if FMTBO_ENABLE_KT:
            self.create_shared_x()

    def create_shared_x(self):
        """Create shared validation set for ranking"""
        size = INITIAL_SAMPLES * DIM 
        self.shared_x = torch.rand(size, DIM, device=DEVICE, dtype=DTYPE) * (bounds[1] - bounds[0]) + bounds[0]

    def aggregate_models(self):
        """
        Perform Model Aggregation.
        If KT is enabled, compute Similarity Matrix using Rank Inversions.
        Reference: deploy/servergp.py -> update_model -> cal_ls_mat
        """
        
        # 1. Collect all client model parameters and Ranks
        client_params = []
        client_ranks = []
        
        for agent in self.agents:
            client_params.append(agent.get_model_params())
            if FMTBO_ENABLE_KT:
                rank = agent.calculate_rank_on_shared(self.shared_x)
                client_ranks.append(rank)
        
        # 2. Aggregation Logic
        if FMTBO_ENABLE_KT:
            n_agents = self.n_agents
            ls_mat = np.zeros((n_agents, n_agents))
            
            # Calculate Similarity Matrix (Inversion Counts)
            for i in range(n_agents):
                rank_i = client_ranks[i]
                for j in range(n_agents):
                    if i == j: continue
                    
                    rank_j = client_ranks[j]
                    
                    mat_i = rank_i[:, None] < rank_i[None, :]
                    mat_j = rank_j[:, None] < rank_j[None, :]
                    inversions = np.logical_xor(mat_i, mat_j)
                    ls_mat[i][j] = np.sum(inversions)

            # Select neighbors and Aggregate
            new_global_params = []
            k = int(n_agents * FMTBO_KT_AGG_PROB)
            
            for i in range(n_agents):
                # Ascend sort (Smallest inversion count = Most similar)
                neighbor_indices = np.argsort(ls_mat[i])[:k]
                selected_params = [client_params[idx] for idx in neighbor_indices]
                averaged = self.fed_avg(selected_params)
                new_global_params.append(averaged)
            
            self.global_model_params = new_global_params
            
        else:
            # Standard FedAvg
            averaged = self.fed_avg(client_params)
            self.global_model_params = [averaged for _ in range(self.n_agents)]

    def fed_avg(self, params_list):
        """Average state_dicts"""
        avg_params = deepcopy(params_list[0])
        n = len(params_list)
        
        for key in avg_params.keys():
            if isinstance(avg_params[key], torch.Tensor):
                sum_tensor = torch.zeros_like(avg_params[key])
                for p in params_list:
                    sum_tensor += p[key]
                avg_params[key] = sum_tensor / n
        return avg_params

    def one_round_execution(self):
        """
        Execute one round of FMTBO:
        1. Clients receive global params (from prev round aggregation).
        2. Clients Fit models (Global + Local) and Select Next Point.
        3. Server Aggregates models for next round.
        """
        
        # 1. Distribute & 2. Client Train (Fit + Select + Obs)
        for i, agent in enumerate(self.agents):
            # Client receives specific global params (if KT)
            agent.fit_models(global_state_dict=self.global_model_params[i])
            agent.select_next_point()
            
        # 3. Server Aggregates (Update Model for next round)
        self.aggregate_models()


def run_single_experiment(exp_id):
    """Run a single FMTBO experiment"""
    print(f"\n=== Running Experiment {exp_id + 1}/{NUM_EXPERIMENTS} ===")
    start_time = time.time()
    
    seed = SEED_BASE + exp_id
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    # Initialization using CLHS
    cubes = equal_CLHS(N_AGENTS, INITIAL_SAMPLES, exp_id)
    
    server = FMTBOServer(N_AGENTS)
    server.initialize_agents(cubes, exp_id)
    
    avg_btv_hist = []
    avg_inst_hist = []
    
    # Initial stats (Iter 0)
    btv_vals = [a.get_btv_value() for a in server.agents]
    inst_vals = [a.get_instantaneous_value() for a in server.agents]
    avg_btv_hist.append(np.mean(btv_vals))
    avg_inst_hist.append(np.mean(inst_vals))
    
    print(f"Iter 0 | Avg BTV: {np.mean(btv_vals):.4f}")
    
    for itr in range(1, MAX_ITERATIONS + 1):
        if ENABLE_DEBUG_PRINT:
            print(f"--- Iter {itr} ---")
            
        # Execute Round
        server.one_round_execution()
        
        # Logging
        btv_vals = [a.get_btv_value() for a in server.agents]
        inst_vals = [a.get_instantaneous_value() for a in server.agents]
        avg_btv_hist.append(np.mean(btv_vals))
        avg_inst_hist.append(np.mean(inst_vals))
        
        print(f"Iter {itr} | Avg BTV: {np.mean(btv_vals):.4f} | Avg Inst: {np.mean(inst_vals):.4f}")
            
    return avg_btv_hist, avg_inst_hist, time.time() - start_time

def main():
    print(f"=== FMTBO Algorithm Experiment ===")
    print(f"Function: {FUNCTION_NAME.upper()}")
    print(f"Dim: {DIM}")
    print(f"FMTBO Config: KT={FMTBO_ENABLE_KT}, Gamma={FMTBO_GAMMA}, AggProb={FMTBO_KT_AGG_PROB}")
    
    all_btv, all_inst, all_time = [], [], []
    
    for i in range(NUM_EXPERIMENTS):
        btv, ins, t = run_single_experiment(i)
        all_btv.append(btv); all_inst.append(ins); all_time.append(t)
        
    # Generate output filename
    output_prefix = f"FMTBO_{FUNCTION_NAME.upper()}{DIM}_S{HETER_SHIFT}_R{HETER_ROTATION}"
    
    # Save CSVs
    iters = list(range(MAX_ITERATIONS + 1))
    
    # Best True Value
    btv_df = pd.DataFrame({'iteration': iters})
    for i in range(NUM_EXPERIMENTS):
        btv_df[f'exp_{i+1}'] = all_btv[i]
    btv_csv = f"{output_prefix}_btv.csv"
    btv_df.to_csv(btv_csv, index=False)
    print(f"\nBest True Value values saved to: {btv_csv}")
    
    # Instantaneous
    inst_df = pd.DataFrame({'iteration': iters})
    for i in range(NUM_EXPERIMENTS):
        inst_df[f'exp_{i+1}'] = all_inst[i]
    inst_csv = f"{output_prefix}_instantaneous.csv"
    inst_df.to_csv(inst_csv, index=False)
    print(f"Instantaneous values saved to: {inst_csv}")
    
    # === Timing Stats ===
    print(f"\n=== Experiment Finished ===")
    print(f"Final Avg Best True Value: {np.mean([r[-1] for r in all_btv]):.4f}")
    
    print(f"\n=== Timing Stats ===")
    timing_lines = []
    for exp_id, elapsed_time in enumerate(all_time):
        timing_line = f"Exp {exp_id + 1} Time: {elapsed_time:.2f} s"
        print(timing_line)
        timing_lines.append(timing_line)
    
    avg_time = np.mean(all_time)
    total_time = np.sum(all_time)
    
    print("---")
    avg_line = f"Avg Time: {avg_time:.2f} s"
    total_line = f"Total Time: {total_time:.2f} s"
    print(avg_line)
    print(total_line)
    
    timing_lines.append("---")
    timing_lines.append(avg_line)
    timing_lines.append(total_line)
    
    with open(f"{output_prefix}_timing.txt", 'w', encoding='utf-8') as f:
        f.write("=== FMTBO Algorithm Timing Stats ===\n")
        for line in timing_lines:
            f.write(line + "\n")
    
    print(f"\nTiming stats saved to: {output_prefix}_timing.txt")

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings('ignore')
    main()