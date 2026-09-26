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
from botorch.acquisition import (AnalyticAcquisitionFunction, 
                                  ExpectedImprovement, LogNoisyExpectedImprovement)
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler, IIDNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition.analytic import _scaled_improvement, _log_ei_helper
from botorch.utils.safe_math import logmeanexp
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning

#Global Configuration

# Device and Data Type
DEVICE = torch.device("cpu")
DTYPE = torch.double

# Target Function Selection 
# 'ackley', 'levy', 'griewank', 'rastrigin', 
# 'rosenbrock', 'sphere', 'weierstrass', 'ellipsoid', 'zakharov', 
# 'michalewicz', 'powell', 'styblinskitang'
FUNCTION_NAME = 'ellipsoid'

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

# Heterogeneity levels (shift, rotation): homogeneous (0, 0), mild (0.05, 0.1), and severe (0.3, 1.0).
HETER_SHIFT = 0.3
HETER_ROTATION = 1.0

# Group size, communication interval, sampling limits, improvement threshold, and LCB beta.
CNEI_MAX_GROUP_SIZE = 4
CNEI_COM_ITR = 1
CNEI_MAX_SAMPLE = 20
CNEI_MIN_SAMPLE = 5
CNEI_TAU = 1e-4
CNEI_LCB_BETA = 2.0

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


# Init blackbox and bounds
blackbox = get_blackbox_function(FUNCTION_NAME, CURRENT_CONFIG)
bounds = torch.tensor(CURRENT_CONFIG['bounds'], device=DEVICE, dtype=DTYPE) / NORMALIZE_X


def truth(X):
    return blackbox(X * NORMALIZE_X) / NORMALIZE_Y


#CLHS Initialization

# Pass the experiment ID to isolate random streams.
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


#CNEI Specific Acquisition Function

class LowerConfidenceBound(AnalyticAcquisitionFunction):
    
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


class LNEI(ExpectedImprovement):
    """Log Noisy Expected Improvement with Fantasy Models"""
    
    def __init__(
        self,
        model_list,
        X_observed,
        num_fantasies: int = 20,
        maximize: bool = True
    ):
        Fantasy_X = torch.tensor([]).to(DEVICE)
        Fantasy_Y = torch.tensor([]).to(DEVICE)
        best_f = torch.tensor([]).to(DEVICE)
        
        for model in model_list:
            train_X = model.train_inputs[0]
            with torch.no_grad():
                posterior = model.posterior(X=train_X)
                sampler = SobolQMCNormalSampler(num_fantasies)
                Y_fantasized = sampler(posterior).squeeze(-1)
            best_f = torch.cat([best_f, Y_fantasized[:, :X_observed.shape[0]].max(dim=-1)[0]])
            Fantasy_X = torch.cat([Fantasy_X, train_X.expand(num_fantasies, *train_X.shape)])
            Fantasy_Y = torch.cat([Fantasy_Y, Y_fantasized])
        
        fantasy_model = FixedNoiseGP(
            train_X=torch.zeros(1, bounds.shape[1]).to(DTYPE),
            train_Y=torch.zeros(1, 1).to(DTYPE),
            train_Yvar=torch.ones(1, 1).to(DTYPE),
        ).to(DEVICE)
        
        fantasy_model.set_train_data(inputs=Fantasy_X, targets=Fantasy_Y, strict=False)
        fantasy_model.likelihood.noise_covar.noise = torch.full_like(Fantasy_Y, SMALL_NUG)
        fantasy_model.load_state_dict(model.state_dict())
        
        super().__init__(model=fantasy_model, best_f=best_f, maximize=maximize)
    
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X.unsqueeze(-3))
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        log_ei = _log_ei_helper(u) + sigma.log()
        return logmeanexp(log_ei, dim=-1)


#CNEI Agent Class

class CNEIAgent:
    """Agent class for CNEI algorithm"""
    
    # Pass the experiment ID to isolate random streams.
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
        
        # CNEI specific attributes
        self.pseudo_x = Tensor([]).to(DEVICE)
        self.have_fantasy = False
        self.truncate_models = None
        self.optimized_param = None
        
        # Sampling points and values
        self.next_sample = None
        self.proposed_design = None
        self.proposed_value = None
        
        # Best value for fantasy models
        self.best_value = -float('inf')
        
        # Performance metrics
        self.instantaneous_value = 0.0
        self.btv_value = -float('inf')
    
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
    
    # Pass the experiment ID to isolate random streams.
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
        
        self._update_best_value()
    
    def _update_best_value(self):
        """Calculate and update the max of GP posterior mean"""
        num_restarts = 20
        raw_samples = 100
        batch_limit = 50
        max_itr = 25

        from botorch.acquisition import PosteriorMean
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
        """Propose design using LCB"""
        num_restarts = 20
        raw_samples = 100
        batch_limit = 50
        max_itr = 25

        criteria = LowerConfidenceBound(self.model, CNEI_LCB_BETA)
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
        patterns = patterns[counts > CNEI_MIN_SAMPLE]
        
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
            print(f"  Agent {self.agent_id}: Excluded designs {(pseudo_x.shape[0] - value[-1]).item()}")
    
    def get_next_design(self):
        """Select next observation point"""
        num_restarts = 20
        raw_samples = 100
        batch_limit = 50
        max_itr = 25

        if self.have_fantasy:
            acq_func = LNEI(
                model_list=self.truncate_models,
                X_observed=self.local_x
            )
        else:
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
        """Execute next observation and update model"""
        new_x = self.next_sample
        new_y = self.observation(new_x)
        self.local_x = torch.cat([self.local_x, new_x])
        self.local_y = torch.cat([self.local_y, new_y])
        self.update_model()
    
    def get_instantaneous_value(self):
        return self.instantaneous_value
    
    def get_btv_value(self):
        return self.btv_value


#CNEI Server Class

class CNEIServer:
    """CNEI Server Class (Manages cooperation)"""
    
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agents = []
        self.grouping = []
    
    # Use the experiment ID for reproducible setup.
    def initialize_agents(self, cubes, exp_id=0):
        """Initialize all agents"""
        for i in range(self.n_agents):
            # Pass the experiment ID to isolate heterogeneity generation.
            agent = CNEIAgent(i, cubes[i][0], cubes[i][1], exp_id)
            # Pass the experiment ID to isolate initial sampling.
            agent.generate_initial_data(exp_id)
            self.agents.append(agent)
    
    def random_grouping(self, input_list):
        """Random grouping"""
        random.shuffle(input_list)
        N_Groups = int(np.ceil(self.n_agents / CNEI_MAX_GROUP_SIZE))
        return [sublist.tolist() for sublist in np.array_split(input_list, N_Groups)]
    
    def conference(self, iteration):
        """Cooperation conference (Exchange info and generate fantasy models)"""
        if (iteration - 1) % CNEI_COM_ITR == 0:
            self.grouping = self.random_grouping(self.agents.copy())
        
        borrow_count = 0
        
        for group in self.grouping:
            best_preds = []
            proposal = torch.tensor([]).to(DEVICE)
            
            for agent in group:
                agent.propose()
                best_preds.append(agent.proposed_value)
                proposal = torch.cat((proposal, agent.proposed_design))
            
            best_preds = np.array(best_preds)
            
            for i, agent in enumerate(group):
                emulation = proposal[best_preds > agent.best_value + CNEI_TAU]
                agent.gen_fantasy_models(emulation, num_fantasies=CNEI_MAX_SAMPLE)
                borrow_count += agent.have_fantasy
        
        if ENABLE_DEBUG_PRINT:
            print(f"  Agents using fantasy models: {borrow_count}")
    
    def researching(self):
        """All agents execute observation"""
        for agent in self.agents:
            agent.get_next_design()
            agent.get_next_observation()


#Main Experiment Function

# Run one seeded experimental repetition.
def run_single_cnei_experiment(exp_id):
    print(f"\n=== Running Experiment {exp_id + 1}/{NUM_EXPERIMENTS} ===")
    
    exp_start_time = time.time()
    
    seed = SEED_BASE + exp_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 1. Pass the experiment ID to common Latin hypercube sampling.
    cubes = equal_CLHS(N_AGENTS, INITIAL_SAMPLES, exp_id)
    
    server = CNEIServer(N_AGENTS)
    # 2. Pass the experiment ID to server initialization.
    server.initialize_agents(cubes, exp_id)
    
    avg_btv_values = []
    avg_instantaneous_values = []
    
    print(f"Iter 0 | Initialization Complete (Strategy: CLHS)")
    
    btv_values = [agent.get_btv_value() for agent in server.agents]
    instantaneous_values = [agent.get_instantaneous_value() for agent in server.agents]
    
    avg_btv = np.mean(btv_values)
    avg_instantaneous = np.mean(instantaneous_values)
    
    avg_btv_values.append(avg_btv)
    avg_instantaneous_values.append(avg_instantaneous)
    
    print(f"Iter 0 | Avg BTV: {avg_btv:.4f} | Avg Instantaneous: {avg_instantaneous:.4f}")
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        if ENABLE_DEBUG_PRINT:
            print(f"\n--- Iter {iteration} ---")
        
        server.conference(iteration)
        
        server.researching()
        
        btv_values = [agent.get_btv_value() for agent in server.agents]
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
    print(f"=== CNEI Algorithm Experiment ===")
    print(f"Function: {FUNCTION_NAME.upper()}")
    print(f"Dim: {DIM}")
    print(f"Theoretical Max: {THEORETICAL_MAX}")
    print(f"Algorithm Config:")
    print(f"  - Init Strategy: CLHS")
    print(f"  - Max Group Size: {CNEI_MAX_GROUP_SIZE}")
    print(f"  - Coop Interval: {CNEI_COM_ITR}")
    print(f"  - LCB Beta: {CNEI_LCB_BETA}")
    print(f"Experiment Params: N_agents={N_AGENTS}, MAX_ITERATIONS={MAX_ITERATIONS}")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    
    all_btv_results = []
    all_instantaneous_results = []
    all_timings = []
    
    for exp_id in range(NUM_EXPERIMENTS):
        btv_vals, instantaneous_vals, elapsed_time = run_single_cnei_experiment(exp_id)
        all_btv_results.append(btv_vals)
        all_instantaneous_results.append(instantaneous_vals)
        all_timings.append(elapsed_time)
    
    output_prefix = f"CNEI_{FUNCTION_NAME.upper()}{DIM}_S{HETER_SHIFT}_R{HETER_ROTATION}"
    
    iterations = list(range(MAX_ITERATIONS + 1))
    
    btv_df = pd.DataFrame({'iteration': iterations})
    for exp_id in range(NUM_EXPERIMENTS):
        btv_df[f'exp_{exp_id + 1}'] = all_btv_results[exp_id]
    btv_csv = f"{output_prefix}_btv.csv"
    btv_df.to_csv(btv_csv, index=False)
    print(f"\nBest-so-far True Values saved to: {btv_csv}")
    
    instantaneous_df = pd.DataFrame({'iteration': iterations})
    for exp_id in range(NUM_EXPERIMENTS):
        instantaneous_df[f'exp_{exp_id + 1}'] = all_instantaneous_results[exp_id]
    instantaneous_csv = f"{output_prefix}_instantaneous.csv"
    instantaneous_df.to_csv(instantaneous_csv, index=False)
    print(f"Instantaneous values saved to: {instantaneous_csv}")
    
    final_avg_btv = np.mean([results[-1] for results in all_btv_results])
    final_avg_instantaneous = np.mean([results[-1] for results in all_instantaneous_results])
    
    print(f"\n=== Experiment Finished ===")
    print(f"Final Avg Best-so-far True Value: {final_avg_btv:.4f}")
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
        f.write("=== CNEI Algorithm Timing Stats ===\n")
        for line in timing_lines:
            f.write(line + "\n")
    
    print(f"\nTiming stats saved to: {timing_txt}")
    
    print(f"\n=== Config Summary ===")
    print(f"Function: {FUNCTION_NAME} | Dim: {DIM} | Init: CLHS")
    print(f"Group Size: {CNEI_MAX_GROUP_SIZE} | Interval: {CNEI_COM_ITR} | Agents: {N_AGENTS}")


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=InputDataWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    main()