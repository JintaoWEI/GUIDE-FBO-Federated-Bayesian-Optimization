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
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.optim.initializers import initialize_q_batch_nonneg
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RFFKernel
import gpytorch.settings as gpts
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
from botorch.generation.sampling import SamplingStrategy
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)


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

# Thompson candidates for distributed exploration.
FTSDE_N_TS_SAMPLES = 1000
# Number of random Fourier features.
FTSDE_M = 500
# Number of search subregions.
FTSDE_P = 4

ENABLE_DEBUG_PRINT = True


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


def sample_matern_weight(dim, M, p_number=2):
    df = 2 * (p_number + 0.5)
    y = torch.randn(dim, M, dtype=DTYPE, device=DEVICE)
    u = torch.distributions.chi2.Chi2(df).sample((M,)).to(dtype=DTYPE, device=DEVICE)
    randn_weights = (y * torch.sqrt(df / u))
    return randn_weights

def divide_subbounds():
    subbounds = []
    
    subbound1 = bounds.clone()
    subbound1[0, 0] = subbound1[:, 0].mean()
    subbound1[0, 1] = subbound1[:, 1].mean()
    subbounds.append(subbound1)
    
    subbound2 = bounds.clone()
    subbound2[1, 0] = subbound2[:, 0].mean()
    subbound2[1, 1] = subbound2[:, 1].mean()
    subbounds.append(subbound2)
    
    subbound3 = bounds.clone()
    subbound3[1, 0] = subbound3[:, 0].mean()
    subbound3[0, 1] = subbound3[:, 1].mean()
    subbounds.append(subbound3)
    
    subbound4 = bounds.clone()
    subbound4[0, 0] = subbound4[:, 0].mean()
    subbound4[1, 1] = subbound4[:, 1].mean()
    subbounds.append(subbound4)
    
    return subbounds

def optimize_acquisition(f, subbound, sub_weights):
    raw_samples = 100
    num_restarts = 20
    max_itr = 25
    
    Xraw = subbound[0] + (subbound[1] - subbound[0]) * torch.rand(raw_samples, DIM).to(DEVICE, DTYPE)
    Yraw = f(Xraw, sub_weights.data)
    
    X = initialize_q_batch_nonneg(Xraw, Yraw, num_restarts).clone()
    X.requires_grad_(True)
    
    optimizer = torch.optim.Adam([X], lr=0.01)
    
    for i in range(max_itr):
        optimizer.zero_grad()
        loss = -f(X, sub_weights.data).sum()
        loss.backward()
        optimizer.step()
        for j, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., j].clamp_(lb, ub)
    
    X_output = X[f(X, sub_weights.data).argmax()]
    
    return f(X_output, sub_weights.data), X_output.detach()


class MaxPosteriorSampling(SamplingStrategy):
    def __init__(
        self,
        model,
        objective=None,
        posterior_transform=None,
        replacement: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        if objective is None:
            objective = IdentityMCObjective()
        elif not isinstance(objective, MCAcquisitionObjective):
            if posterior_transform is not None:
                assert False
            else:
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=objective.weights, offset=objective.offset
                )
                objective = IdentityMCObjective()
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement
    
    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)
        values, idcs = torch.max(obj, dim=-1)
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        return values[0].detach().reshape((-1, 1)), torch.gather(Xe, -2, idcs)


# Local optimization agent for FTSDE.
class FTSDEAgent:
    def __init__(self, agent_id, assigned_bound, exp_id=0):
        self.agent_id = agent_id
        self.assigned_bound = assigned_bound
        
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
        self.lengthscale = None
        self.outputscale = None
        self.obs_noise = None
        self.rff_kernel = None
        self.borrow_weights = None
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

        self.local_x = torch.rand(INITIAL_SAMPLES, bounds.shape[1], device=DEVICE, dtype=DTYPE)
        self.local_x = unnormalize(self.local_x, self.assigned_bound)
        
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
        self.lengthscale = self.model.covar_module.base_kernel.lengthscale
        self.outputscale = self.model.covar_module.outputscale
        self.obs_noise = NOISE_SE ** 2
        
        self.rff_kernel = RFFKernel(ard_num_dims=DIM, num_samples=FTSDE_M).to(
            dtype=self.lengthscale.dtype,
            device=self.lengthscale.device
        )
    
    def sample_w(self):
        Phi = self.outputscale.sqrt() * self.rff_kernel._featurize(self.local_x, normalize=True)
        
        Sigma_t = (Phi.T @ Phi) + self.obs_noise * torch.eye(2 * FTSDE_M).to(DEVICE).to(DTYPE)
        Sigma_t_inv = torch.linalg.inv(Sigma_t)
        nu_t = ((Sigma_t_inv @ Phi.T) @ self.local_y.reshape(-1, 1))
        
        w_sample = torch.distributions.MultivariateNormal(
            torch.squeeze(nu_t),
            self.obs_noise * Sigma_t_inv
        ).sample()
        
        return w_sample
    
    def predict(self, x, w=None):
        if w is None:
            w = self.sample_w()
        
        features = self.outputscale.sqrt().data * self.rff_kernel._featurize(x, normalize=True)
        features = features.reshape(-1, 2 * FTSDE_M)
        
        f_value = (w @ features.T).squeeze()
        
        return f_value
    
    def get_next_design_ts(self, n_candidates=10000):
        sobol = SobolEngine(self.local_x.shape[-1], scramble=True)
        X_cand = sobol.draw(n_candidates).to(dtype=DTYPE, device=DEVICE)
        X_cand = unnormalize(X_cand, bounds)
        
        with ExitStack() as es:
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            values, candidates = thompson_sampling(X_cand, num_samples=1)
        
        return candidates.detach()
    
    def col_sample(self, subbounds):
        values_list = torch.tensor([]).to(DEVICE)
        candidates_list = torch.tensor([]).to(DEVICE)
        itr = int(N_AGENTS / FTSDE_P)
        
        for i in range(FTSDE_P):
            bias = [*range(itr * i, itr * (i + 1))]
            
            a = 16
            T = (i + 1) ** 2
            
            weights = torch.zeros(N_AGENTS).type(DTYPE).to(DEVICE)
            weights[bias] = 1
            weights = torch.exp((a * weights + 1) / T)
            weights = weights / weights.sum()
            
            sub_weights = weights @ self.borrow_weights
            
            value, candidate = optimize_acquisition(self.predict, subbounds[i], sub_weights)
            values_list = torch.cat([values_list, value.unsqueeze(0)])
            candidates_list = torch.cat([candidates_list, candidate.unsqueeze(0)])
        
        return candidates_list[values_list.argmax()].unsqueeze(0)
    
    def get_next_observation(self):
        new_x = self.next_sample
        new_y = self.observation(new_x)
        self.local_x = torch.cat([self.local_x, new_x])
        self.local_y = torch.cat([self.local_y, new_y])
        self.update_model()
    
    def get_instantaneous_value(self):
        return self.instantaneous_value
    
    def get_btv_value(self):
        return self.btv_value


# Coordinates agent messages for FTSDE.
class FTSDEServer:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agents = []
        self.weight_list = None
        self.subbounds = divide_subbounds()
        self.randn_weights = None
    
    def initialize_agents(self, exp_id=0):
        itr = int(self.n_agents / FTSDE_P)
        
        for i in range(self.n_agents):
            region_idx = i // itr
            assigned_bound = self.subbounds[region_idx]
            
            # Pass the experiment ID to the agent.
            agent = FTSDEAgent(i, assigned_bound, exp_id)
            agent.generate_initial_data(exp_id)
            self.agents.append(agent)
    
    def conference(self):
        self.weight_list = torch.tensor([]).to(DEVICE)
        for agent in self.agents:
            agent.rff_kernel.randn_weights = self.randn_weights
            self.weight_list = torch.cat([self.weight_list, agent.sample_w().unsqueeze(0)])
        
        for agent in self.agents:
            agent.borrow_weights = self.weight_list
    
    def researching(self, iteration):
        p = 1 - 1 / np.sqrt(iteration) if iteration > 1 else 0.5
        
        for agent in self.agents:
            if torch.rand(1).item() <= p:
                agent.next_sample = agent.get_next_design_ts(FTSDE_N_TS_SAMPLES)
            else:
                agent.next_sample = agent.col_sample(self.subbounds)
        
        for agent in self.agents:
            agent.get_next_observation()
        
        if ENABLE_DEBUG_PRINT:
            print(f"  Adaptive probability p = {p:.4f}")


# Run one seeded experimental repetition.
def run_single_ftsde_experiment(exp_id):
    print(f"\n=== Running Experiment {exp_id + 1}/{NUM_EXPERIMENTS} ===")
    exp_start_time = time.time()
    
    seed = SEED_BASE + exp_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    randn_weights = sample_matern_weight(DIM, FTSDE_M)
    
    server = FTSDEServer(N_AGENTS)
    server.randn_weights = randn_weights
    # Pass the experiment ID for reproducibility.
    server.initialize_agents(exp_id)
    
    avg_btv_values = []
    avg_instantaneous_values = []
    
    print(f"Iter 0 | Initialization Complete (Strategy: Sub-region Uniform Sampling)")
    
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
        
        server.conference()
        server.researching(iteration)
        
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
    print(f"=== FTSDE Algorithm Experiment ===")
    print(f"Function: {FUNCTION_NAME.upper()}")
    print(f"Dim: {DIM}")
    print(f"Theoretical Max: {THEORETICAL_MAX}")
    print(f"Algorithm Config:")
    print(f"  - Initialization Strategy: Sub-region Uniform Sampling (FTSDE original)")
    print(f"  - RFF Features: {FTSDE_M}")
    print(f"  - Sub-regions: {FTSDE_P}")
    print(f"  - TS Candidates: {FTSDE_N_TS_SAMPLES}")
    print(f"Experiment Params: N_agents={N_AGENTS}, MAX_ITERATIONS={MAX_ITERATIONS}")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    
    all_btv_results = []
    all_instantaneous_results = []
    all_timings = []
    
    for exp_id in range(NUM_EXPERIMENTS):
        btv_vals, instantaneous_vals, elapsed_time = run_single_ftsde_experiment(exp_id)
        all_btv_results.append(btv_vals)
        all_instantaneous_results.append(instantaneous_vals)
        all_timings.append(elapsed_time)
    
    output_prefix = f"FTSDE_{FUNCTION_NAME.upper()}{DIM}_S{HETER_SHIFT}_R{HETER_ROTATION}"
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
        f.write("=== FTSDE Algorithm Timing Stats ===\n")
        for line in timing_lines:
            f.write(line + "\n")
    
    print(f"\nTiming stats saved to: {timing_txt}")
    print(f"\n=== Config Summary ===")
    print(f"Function: {FUNCTION_NAME} | Dim: {DIM} | Init: Sub-region Uniform")
    print(f"RFF Features: {FTSDE_M} | Sub-regions: {FTSDE_P} | Agents: {N_AGENTS}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=InputDataWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    main()