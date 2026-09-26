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
from botorch.acquisition import PosteriorMean, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.transforms import t_batch_mode_transform, unnormalize
from botorch.generation import MaxPosteriorSampling
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch.settings as gpts
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning

#Global Configuration

# Device and Data Type
DEVICE = torch.device("cpu")
DTYPE = torch.double

# Function Selection
# 'ackley', 'levy', 'griewank', 'rastrigin',
# 'rosenbrock', 'sphere', 'weierstrass', 'ellipsoid', 'zakharov', 
# 'michalewicz', 'powell', 'styblinskitang'
FUNCTION_NAME = 'ellipsoid'  

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

# Group size, communication interval, sampling limits, improvement threshold, and LCB scale.
CTS_N_TS_SAMPLES = 1000  # Thompson Sampling candidates
CTS_MAX_GROUP_SIZE = 4
CTS_COM_ITR = 1
CTS_MAX_SAMPLE = 20
CTS_MIN_SAMPLE = 5
CTS_TAU = 1e-4
CTS_LCB_C = 2.0

# Debug switch
ENABLE_DEBUG_PRINT = True

#Custom Test Functions

class Sphere:
    """Sphere function"""
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
    """Weierstrass function"""
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
    """Ellipsoid function"""
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
    """Zakharov function"""
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


# Get blackbox function
def get_blackbox_function(function_name, config):
    """Get the corresponding blackbox function by name"""
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


# Initialize blackbox and bounds
blackbox = get_blackbox_function(FUNCTION_NAME, CURRENT_CONFIG)
bounds = torch.tensor(CURRENT_CONFIG['bounds'], device=DEVICE, dtype=DTYPE) / NORMALIZE_X


def truth(X):
    """True function evaluation, applying normalization"""
    return blackbox(X * NORMALIZE_X) / NORMALIZE_Y


#CLHS Initialization

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

#CTS Specific Acquisition Function

class LowerConfidenceBound(AnalyticAcquisitionFunction):
    """Lower Confidence Bound Acquisition Function (for proposing designs)"""
    
    def __init__(
        self,
        model,
        beta,
        posterior_transform=None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X)
        return (mean if self.maximize else -mean) - self.beta * sigma


#CTS Agent Class

class CTSAgent:
    """Agent class for CTS Algorithm"""
    
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
        self.model = None
        self.mll = None
        
        # CTS specific attributes
        self.pseudo_x = Tensor([]).to(DEVICE)
        self.have_fantasy = False
        self.truncate_models = None
        self.optimized_param = None
        
        # Sampling points and values
        self.next_sample = None
        self.proposed_design = None
        self.proposed_value = None
        
        # Best value from GP posterior mean (for fantasy models)
        self.best_value = -float('inf')
        
        # Performance metrics
        self.instantaneous_value = 0.0
        self.btv_value = -float('inf')
    
    def observation(self, X):
        """Observe function value (with noise), and store true value"""
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
        
        self.update_model()
        
        # RNG state isolation ends.
        torch.set_rng_state(current_rng_state)
    
    def update_model(self):
        """Update GP model"""
        noise_var = (NOISE_SE ** 2) * torch.ones_like(self.local_y)
        self.model = FixedNoiseGP(self.local_x, self.local_y, noise_var).to(DEVICE)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(DEVICE)
        fit_gpytorch_model(self.mll)
        self.optimized_param = deepcopy(self.model.state_dict())
        
        # Calculate max posterior mean (for fantasy models, position not saved)
        self._update_best_value()
    
    def _update_best_value(self):
        """Calculate and update the maximum of GP posterior mean (only value, not position)"""
        num_restarts = 20
        raw_samples = 100
        batch_limit = 50
        max_itr = 25
        
        criteria = PosteriorMean(self.model)
        candidates, value = optimize_acqf(
            acq_function=criteria,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": batch_limit, "maxiter": max_itr},
        )
        
        self.best_value = value.item()
    
    def propose(self):
        """Propose design (using LCB)"""
        num_restarts = 20
        raw_samples = 100
        batch_limit = 50
        max_itr = 25
        
        criteria = LowerConfidenceBound(self.model, CTS_LCB_C)
        candidates, value = optimize_acqf(
            acq_function=criteria,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": batch_limit, "maxiter": max_itr},
        )
        
        self.proposed_design = candidates
        self.proposed_value = value.item()
    
    def gen_fantasy_models(self, pseudo_x, num_fantasies=20):
        """Generate fantasy models"""
        self.pseudo_x = pseudo_x.clone()
        
        if self.pseudo_x.shape[0] == 0:
            self.have_fantasy = False
            self.truncate_models = None
            return
        
        num_samples = int(1e5)
        sampler = IIDNormalSampler(int(num_samples))
        posterior = self.model.posterior(self.pseudo_x)
        samples = sampler(posterior)
        matching = (samples > self.best_value)
        
        patterns, counts = torch.unique(matching, dim=0, return_counts=True)
        patterns = patterns[counts > CTS_MIN_SAMPLE]
        
        value, indices = patterns.sum(dim=1).squeeze().sort()
        
        if indices.numel() == 1:
            self.have_fantasy = False
            self.truncate_models = None
            return
        
        auto_select = patterns[indices[-1]].reshape(-1)
        self.pseudo_x = pseudo_x[auto_select].clone()
        
        auto_select_exp = auto_select.expand_as(matching.squeeze())
        selected_indices = (matching.squeeze() == auto_select_exp).reshape(num_samples, -1).all(dim=1)
        selected_samples = samples[selected_indices][:, auto_select, :][:num_fantasies]
        
        model_list = []
        
        for fantasy_y in selected_samples:
            train_x = torch.cat([self.local_x, self.pseudo_x])
            train_y = torch.cat([self.local_y, fantasy_y.reshape(-1, 1)])
            noise = torch.cat([
                (NOISE_SE ** 2) * torch.ones_like(self.local_y),
                SMALL_NUG * torch.ones_like(fantasy_y.reshape(-1, 1))
            ])
            
            truncate_model = FixedNoiseGP(train_x, train_y, noise).to(DEVICE)
            truncate_model.load_state_dict(self.optimized_param)
            model_list.append(truncate_model)
        
        self.truncate_models = model_list
        self.have_fantasy = True
        
        if ENABLE_DEBUG_PRINT and value[-1] < pseudo_x.shape[0]:
            print(f"  Agent {self.agent_id}: Excluded designs count {(pseudo_x.shape[0] - value[-1]).item()}")
    
    def get_next_design(self, n_candidates=10000):
        """Select next observation point using Thompson Sampling"""
        # Use Sobol sequence to generate candidates
        sobol = SobolEngine(self.local_x.shape[-1], scramble=True)
        X_cand = sobol.draw(n_candidates).to(dtype=DTYPE, device=DEVICE)
        X_cand = unnormalize(X_cand, bounds)
        
        # Select model to use
        if self.have_fantasy:
            _model = self.truncate_models[0]
        else:
            _model = self.model
        
        # Thompson Sampling
        with ExitStack() as es:
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            thompson_sampling = MaxPosteriorSampling(model=_model, replacement=False)
            candidates = thompson_sampling(X_cand, num_samples=1)
        
        self.next_sample = candidates.detach()
    
    def get_next_observation(self):
        """Execute next observation and update model"""
        new_x = self.next_sample
        new_y = self.observation(new_x)
        self.local_x = torch.cat([self.local_x, new_x])
        self.local_y = torch.cat([self.local_y, new_y])
        self.update_model()
    
    def get_instantaneous_value(self):
        """Return instantaneous value (true value of current point)"""
        return self.instantaneous_value
    
    def get_btv_value(self):
        """Return best-so-far true value"""
        return self.btv_value


#CTS Server Class

class CTSServer:
    """Server class for CTS Algorithm (Manages collaboration)"""
    
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agents = []
        self.grouping = []
    
    # Use the experiment ID for reproducible setup.
    def initialize_agents(self, cubes, exp_id=0):
        """Initialize all agents"""
        for i in range(self.n_agents):
            # Pass the experiment ID to isolate heterogeneity generation.
            agent = CTSAgent(i, cubes[i][0], cubes[i][1], exp_id)
            # Pass the experiment ID to isolate initial sampling.
            agent.generate_initial_data(exp_id)
            self.agents.append(agent)
    
    def random_grouping(self, input_list):
        """Random grouping"""
        random.shuffle(input_list)
        N_Groups = int(np.ceil(self.n_agents / CTS_MAX_GROUP_SIZE))
        return [sublist.tolist() for sublist in np.array_split(input_list, N_Groups)]
    
    def conference(self, iteration):
        """Collaboration conference (Exchange info and generate fantasy models)"""
        # Regroup every Com_itr rounds
        if (iteration - 1) % CTS_COM_ITR == 0:
            self.grouping = self.random_grouping(self.agents.copy())
        
        borrow_count = 0
        
        for group in self.grouping:
            best_preds = []
            proposal = torch.tensor([]).to(DEVICE)
            
            # Collect proposals from each agent
            for agent in group:
                agent.propose()
                best_preds.append(agent.proposed_value)
                proposal = torch.cat((proposal, agent.proposed_design))
            
            best_preds = np.array(best_preds)
            
            # Generate fantasy models for each agent (using best_value instead of best_observed_value)
            for i, agent in enumerate(group):
                emulation = proposal[best_preds > agent.best_value + CTS_TAU]
                agent.gen_fantasy_models(emulation, num_fantasies=CTS_MAX_SAMPLE)
                borrow_count += agent.have_fantasy
        
        if ENABLE_DEBUG_PRINT:
            print(f"  Number of agents using fantasy models: {borrow_count}")
    
    def researching(self):
        """All agents execute observations"""
        for agent in self.agents:
            agent.get_next_design(CTS_N_TS_SAMPLES)
            agent.get_next_observation()


#Main Experiment Function

def run_single_cts_experiment(exp_id):
    """Run a single CTS experiment"""
    print(f"\n=== Running Experiment {exp_id + 1}/{NUM_EXPERIMENTS} ===")
    
    # Record experiment start time
    exp_start_time = time.time()
    
    # Set random seed
    seed = SEED_BASE + exp_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 1. Pass the experiment ID to common Latin hypercube sampling.
    cubes = equal_CLHS(N_AGENTS, INITIAL_SAMPLES, exp_id)
    
    # Initialize server and agents
    server = CTSServer(N_AGENTS)
    server.initialize_agents(cubes, exp_id)
    
    # Record average metrics per iteration
    avg_btv_values = []
    avg_instantaneous_values = []
    
    print(f"Iter 0 | Initialization done (Strategy: CLHS)")
    
    # Record initial state (Iter 0)
    btv_values = [agent.get_btv_value() for agent in server.agents]
    instantaneous_values = [agent.get_instantaneous_value() for agent in server.agents]
    
    avg_btv = np.mean(btv_values)
    avg_instantaneous = np.mean(instantaneous_values)
    
    avg_btv_values.append(avg_btv)
    avg_instantaneous_values.append(avg_instantaneous)
    
    print(f"Iter 0 | Avg BTV: {avg_btv:.4f} | Avg Instantaneous: {avg_instantaneous:.4f}")
    
    # Main Loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        if ENABLE_DEBUG_PRINT:
            print(f"\n--- Iter {iteration} ---")
        
        # Collaboration conference
        server.conference(iteration)
        
        # Execute research/observation
        server.researching()
        
        # Calculate average metrics for current iteration
        btv_values = [agent.get_btv_value() for agent in server.agents]
        instantaneous_values = [agent.get_instantaneous_value() for agent in server.agents]
        
        avg_btv = np.mean(btv_values)
        avg_instantaneous = np.mean(instantaneous_values)
        
        avg_btv_values.append(avg_btv)
        avg_instantaneous_values.append(avg_instantaneous)
        
        if ENABLE_DEBUG_PRINT:
            print(f"Iter {iteration} | Avg BTV: {avg_btv:.4f} | Avg Instantaneous: {avg_instantaneous:.4f}")
    
    # Calculate experiment elapsed time
    exp_elapsed_time = time.time() - exp_start_time
    
    return avg_btv_values, avg_instantaneous_values, exp_elapsed_time


def main():
    """Main Function"""
    print(f"=== CTS Algorithm Experiment ===")
    print(f"Function: {FUNCTION_NAME.upper()}")
    print(f"Dim: {DIM}")
    print(f"Theoretical Max: {THEORETICAL_MAX}")
    print(f"Algorithm Config:")
    print(f"  - Init Strategy: CLHS")
    print(f"  - Max Group Size: {CTS_MAX_GROUP_SIZE}")
    print(f"  - Communication Interval: {CTS_COM_ITR}")
    print(f"  - LCB C: {CTS_LCB_C}")
    print(f"  - TS Candidates: {CTS_N_TS_SAMPLES}")
    print(f"Exp Params: N_agents={N_AGENTS}, MAX_ITERATIONS={MAX_ITERATIONS}")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    
    # Store results for all experiments
    all_btv_results = []
    all_instantaneous_results = []
    all_timings = []
    
    # Run multiple experiments
    for exp_id in range(NUM_EXPERIMENTS):
        btv_vals, instantaneous_vals, elapsed_time = run_single_cts_experiment(exp_id)
        all_btv_results.append(btv_vals)
        all_instantaneous_results.append(instantaneous_vals)
        all_timings.append(elapsed_time)
    
    # Generate output filename
    output_prefix = f"CTS_{FUNCTION_NAME.upper()}{DIM}_S{HETER_SHIFT}_R{HETER_ROTATION}"
    
    # Create DataFrame and save to CSV
    iterations = list(range(MAX_ITERATIONS + 1))
    
    # BTV Value CSV
    btv_df = pd.DataFrame({'iteration': iterations})
    for exp_id in range(NUM_EXPERIMENTS):
        btv_df[f'exp_{exp_id + 1}'] = all_btv_results[exp_id]
    btv_csv = f"{output_prefix}_btv.csv"
    btv_df.to_csv(btv_csv, index=False)
    print(f"\nBTV values saved to: {btv_csv}")
    
    # Instantaneous Value CSV
    instantaneous_df = pd.DataFrame({'iteration': iterations})
    for exp_id in range(NUM_EXPERIMENTS):
        instantaneous_df[f'exp_{exp_id + 1}'] = all_instantaneous_results[exp_id]
    instantaneous_csv = f"{output_prefix}_instantaneous.csv"
    instantaneous_df.to_csv(instantaneous_csv, index=False)
    print(f"Instantaneous values saved to: {instantaneous_csv}")
    
    # Output final results summary
    final_avg_btv = np.mean([results[-1] for results in all_btv_results])
    final_avg_instantaneous = np.mean([results[-1] for results in all_instantaneous_results])
    
    print(f"\n=== Experiment Finished ===")
    print(f"Final Avg Best-so-far True Value: {final_avg_btv:.4f}")
    print(f"Final Avg Instantaneous Value: {final_avg_instantaneous:.4f}")
    
    # Output and save timing stats
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
    
    # Save timing results
    timing_txt = f"{output_prefix}_timing.txt"
    with open(timing_txt, 'w', encoding='utf-8') as f:
        f.write("=== CTS Algorithm Timing Stats ===\n")
        for line in timing_lines:
            f.write(line + "\n")
    
    print(f"\nTiming stats saved to: {timing_txt}")
    
    print(f"\n=== Config Summary ===")
    print(f"Function: {FUNCTION_NAME} | Dim: {DIM} | Init: CLHS")
    print(f"Group Size: {CTS_MAX_GROUP_SIZE} | Com Interval: {CTS_COM_ITR} | Agents: {N_AGENTS}")


if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=InputDataWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    main()