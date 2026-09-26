"""
GUIDE-FBO utility functions
Provides test functions, coordinated initialization, and GMM operations,
as well as the federated interventional GP and continuous Thompson sampling.
"""
import torch
import math
import copy
import numpy as np
from torch import Tensor
from botorch.test_functions import (
    Ackley, Levy, Rosenbrock,
    Griewank, Rastrigin, StyblinskiTang, 
    Michalewicz, Powell
)
from botorch.acquisition import AcquisitionFunction
from botorch.sampling.normal import SobolQMCNormalSampler
from sklearn.mixture import BayesianGaussianMixture
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel

from GUIDE_Config import *

# ================== Benchmark functions ==================
class Sphere:
    def __init__(self, dim=10, negate=False):
        self.dim = dim; self.negate = negate
    def __call__(self, X):
        if X.dim() == 1: X = X.unsqueeze(0)
        result = torch.sum(X**2, dim=-1)
        if self.negate: result = -result
        return result.squeeze() if result.numel() == 1 else result

class Weierstrass:
    def __init__(self, dim=10, negate=False):
        self.dim = dim; self.negate = negate
    def __call__(self, X):
        if X.dim() == 1: X = X.unsqueeze(0)
        a, b, kmax = 0.5, 3, 21
        obj = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1]):
            for k in range(kmax):
                obj += (a**k) * torch.cos(2 * torch.pi * (b**k) * (X[..., i] + 0.5))
        constant_term = torch.tensor(0.5, device=X.device, dtype=X.dtype)
        for k in range(kmax):
            obj -= X.shape[-1] * (a**k) * torch.cos(2 * torch.pi * (b**k) * constant_term)
        if self.negate: obj = -obj
        return obj.squeeze() if obj.numel() == 1 else obj

class Ellipsoid:
    def __init__(self, dim=10, negate=False):
        self.dim = dim; self.negate = negate
    def __call__(self, X):
        if X.dim() == 1: X = X.unsqueeze(0)
        weights = torch.arange(1, X.shape[-1] + 1, device=X.device, dtype=X.dtype)
        result = torch.sum(weights * X**2, dim=-1)
        if self.negate: result = -result
        return result.squeeze() if result.numel() == 1 else result

class Zakharov:
    def __init__(self, dim=10, negate=False):
        self.dim = dim; self.negate = negate
    def __call__(self, X):
        if X.dim() == 1: X = X.unsqueeze(0)
        term1 = torch.sum(X**2, dim=-1)
        indices = torch.arange(1, X.shape[-1] + 1, device=X.device, dtype=X.dtype)
        weighted_sum = torch.sum(0.5 * indices * X, dim=-1)
        result = term1 + weighted_sum**2 + weighted_sum**4
        if self.negate: result = -result
        return result.squeeze() if result.numel() == 1 else result

# Construct the selected synthetic benchmark objective.
def get_blackbox_function(function_name, config):
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
    else: raise ValueError(f"Unsupported benchmark function: {function_name}")

blackbox = get_blackbox_function(FUNCTION_NAME, CURRENT_CONFIG)
bounds = torch.tensor(CURRENT_CONFIG['bounds'], device=DEVICE, dtype=DTYPE) / NORMALIZE_X

def truth(X):
    return blackbox(X * NORMALIZE_X) / NORMALIZE_Y

# ================== Initialization strategy ==================
def equal_CLHS(n_agents, initial_samples, exp_id=0):
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(SEED_BASE + exp_id * 10000)
    agents_cubes = [[None, None] for _ in range(n_agents)]
    for dim in range(bounds.shape[1]):
        lb, ub = bounds[:, dim]
        divide_cube = torch.linspace(lb, ub, n_agents * initial_samples + 1)
        for i, agent_id in enumerate(torch.randperm(n_agents)):
            LHS_perm = torch.randperm(initial_samples)
            cube_length_value = (ub - lb) / (n_agents * initial_samples)
            pts = divide_cube[i:-1:n_agents][LHS_perm].unsqueeze(-1)
            lens = Tensor([cube_length_value])
            if agents_cubes[agent_id][0] is None:
                agents_cubes[agent_id][0], agents_cubes[agent_id][1] = pts, lens
            else:
                agents_cubes[agent_id][0] = torch.hstack([agents_cubes[agent_id][0], pts])
                agents_cubes[agent_id][1] = torch.hstack([agents_cubes[agent_id][1], lens])
    torch.set_rng_state(current_rng_state)
    return [(c[0], c[1]) for c in agents_cubes]

# ================== Distribution utilities ==================
def apply_weight_reset(weights, strategy_values):
    min_temperature = 1e-2
    weights, strategy_values = np.array(weights), np.array(strategy_values)
    val_range = np.max(strategy_values) - np.min(strategy_values)
    temp = max(0.1 * val_range, min_temperature) if val_range > 1e-6 else min_temperature
    exp_values = np.exp(strategy_values / temp)
    weighted_exp = weights * exp_values
    return (weighted_exp / np.sum(weighted_exp)).tolist()

def get_beta_value():
    return FIXED_BETA

def fit_dpgmm(points):
    if points.shape[0] == 0: return {'weights': [], 'means': [], 'covariances': []}
    points_np = points.detach().cpu().numpy()
    if len(np.unique(points_np, axis=0)) < len(points_np) * 0.7:
        points_np += np.random.normal(0, 1e-6, points_np.shape)
    actual_components = min(DIM, points_np.shape[0])
    for reg_covar in [1e-6, 1e-5, 1e-4]:
        try:
            dpgmm = BayesianGaussianMixture(
                n_components=actual_components, 
                covariance_type=DPGMM_COVARIANCE_TYPE,
                weight_concentration_prior=100.0, mean_precision_prior=1e-3,
                reg_covar=reg_covar, max_iter=200, n_init=3
            )
            dpgmm.fit(points_np)
            if dpgmm.converged_:
                valid = dpgmm.weights_ > (1.0 / actual_components * 0.01)
                if not np.any(valid): valid[np.argmax(dpgmm.weights_)] = True
                # scikit-learn returns a covariance for the selected mode; diagonal mode stores d variances.
                covs = np.asarray(dpgmm.covariances_[valid])
                return {
                    'weights': dpgmm.weights_[valid].tolist(),
                    'means': dpgmm.means_[valid].tolist(),
                    'covariances': covs.tolist()
                }
        except: continue
    mean = np.mean(points_np, axis=0)
    if DPGMM_COVARIANCE_TYPE == 'diag':
        cov = np.clip(np.var(points_np, axis=0) + 1e-4, 1e-10, None)
    else:
        cov = np.atleast_2d(np.cov(points_np, rowvar=False))
        if cov.shape != (points_np.shape[1], points_np.shape[1]):
            cov = np.eye(points_np.shape[1]) * 1e-4
        else:
            cov = 0.5 * (cov + cov.T) + np.eye(points_np.shape[1]) * 1e-4
    return {'weights': [1.0], 'means': [mean.tolist()], 'covariances': [cov.tolist()]}


def fit_single_gaussian(points):
    """Fit one diagonal Gaussian to all posterior optimum-location samples."""
    if points.shape[0] == 0:
        return {'weights': [], 'means': [], 'covariances': []}

    points_np = points.detach().cpu().numpy()
    mean = np.mean(points_np, axis=0)
    # Reuse the diagonal DPGMM fallback regularization and covariance floor.
    covariance = np.clip(np.var(points_np, axis=0) + 1e-4, 1e-10, None)
    return {
        'weights': [1.0],
        'means': [mean.tolist()],
        'covariances': [covariance.tolist()],
    }

FULL_COVARIANCE_JITTER_EVENTS = 0


def _source_agent_ids(distribution):
    """Return a deterministic, sorted source-agent list for one component."""
    source_ids = distribution.get('source_agent_ids')
    if source_ids is None:
        agent_id = distribution.get('agent_id')
        source_ids = [] if agent_id in (None, 'merged') else [agent_id]
    return sorted(set(source_ids), key=lambda value: str(value))


def _validate_distribution_shape(distribution, covariance_mode):
    mean = np.asarray(distribution['mean'], dtype=float)
    covariance = np.asarray(distribution['covariance'], dtype=float)
    if mean.ndim != 1:
        raise ValueError(f"Distribution mean must have shape (d,), got {mean.shape}.")
    expected = (mean.size,) if covariance_mode == 'diag' else (mean.size, mean.size)
    if covariance.shape != expected:
        raise ValueError(
            f"{covariance_mode} covariance must have shape {expected}, got {covariance.shape}."
        )
    return mean, covariance


def bhattacharyya_distance_diag(mean_a, covariance_a, mean_b, covariance_b):
    """Bhattacharyya distance between diagonal Gaussians using variance vectors."""
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    variance_a = np.asarray(covariance_a, dtype=float)
    variance_b = np.asarray(covariance_b, dtype=float)
    if mean_a.ndim != 1 or mean_b.shape != mean_a.shape:
        raise ValueError("Both means must be one-dimensional vectors with identical shape.")
    if variance_a.ndim != 1 or variance_b.ndim != 1:
        raise ValueError("Diagonal covariance inputs must be one-dimensional variance vectors.")
    if variance_a.shape != mean_a.shape or variance_b.shape != mean_a.shape:
        raise ValueError("Mean and diagonal variance vectors must have identical shape.")
    variance_a = np.clip(variance_a, 1e-10, None)
    variance_b = np.clip(variance_b, 1e-10, None)
    variance_bar = 0.5 * (variance_a + variance_b)
    delta = mean_a - mean_b
    mean_term = 0.125 * np.sum(delta ** 2 / variance_bar)
    covariance_term = 0.5 * np.sum(
        np.log(variance_bar / np.sqrt(variance_a * variance_b))
    )
    return float(max(0.0, mean_term + covariance_term))


def _cholesky_with_jitter(covariance):
    """Return a stable Cholesky factor and count any required jitter event."""
    global FULL_COVARIANCE_JITTER_EVENTS
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    identity = np.eye(covariance.shape[0])
    for jitter in (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        try:
            factor = np.linalg.cholesky(covariance + jitter * identity)
            if jitter > 0.0:
                FULL_COVARIANCE_JITTER_EVENTS += 1
            return factor
        except np.linalg.LinAlgError:
            continue
    raise ValueError("Full covariance is not positive definite after jitter up to 1e-3.")


def reset_full_covariance_jitter_events():
    global FULL_COVARIANCE_JITTER_EVENTS
    FULL_COVARIANCE_JITTER_EVENTS = 0


def get_full_covariance_jitter_events():
    return FULL_COVARIANCE_JITTER_EVENTS


def bhattacharyya_distance_full(mean_a, covariance_a, mean_b, covariance_b):
    """Stable Bhattacharyya distance between full-covariance Gaussians."""
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    covariance_a = np.asarray(covariance_a, dtype=float)
    covariance_b = np.asarray(covariance_b, dtype=float)
    if mean_a.ndim != 1 or mean_b.shape != mean_a.shape:
        raise ValueError("Both means must be one-dimensional vectors with identical shape.")
    expected = (mean_a.size, mean_a.size)
    if covariance_a.shape != expected or covariance_b.shape != expected:
        raise ValueError(f"Full covariance inputs must both have shape {expected}.")
    covariance_bar = 0.5 * (covariance_a + covariance_b)
    chol_a = _cholesky_with_jitter(covariance_a)
    chol_b = _cholesky_with_jitter(covariance_b)
    chol_bar = _cholesky_with_jitter(covariance_bar)
    delta = mean_a - mean_b
    solved = np.linalg.solve(chol_bar, delta)
    mean_term = 0.125 * np.dot(solved, solved)
    logdet_a = 2.0 * np.sum(np.log(chol_a.diagonal()))
    logdet_b = 2.0 * np.sum(np.log(chol_b.diagonal()))
    logdet_bar = 2.0 * np.sum(np.log(chol_bar.diagonal()))
    covariance_term = 0.5 * (logdet_bar - 0.5 * (logdet_a + logdet_b))
    return float(max(0.0, mean_term + covariance_term))


def rms_euclidean_distance(mean_a, mean_b):
    """Return Euclidean distance normalized by the square root of dimension."""
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    if mean_a.ndim != 1 or mean_b.shape != mean_a.shape:
        raise ValueError("Both means must be one-dimensional vectors with identical shape.")
    if mean_a.size == 0:
        raise ValueError("Mean vectors must not be empty.")
    return float(np.sqrt(np.mean((mean_a - mean_b) ** 2)))


def diagonal_moment_match_cluster(distributions):
    """Moment-match one final cluster into a single diagonal Gaussian component."""
    components = list(distributions)
    if not components:
        raise ValueError("Cannot moment-match an empty distribution cluster.")

    validated = [_validate_distribution_shape(component, 'diag') for component in components]
    means = np.stack([mean for mean, _ in validated])
    variances = np.stack([variance for _, variance in validated])
    weights = np.asarray([float(component['weight']) for component in components])
    weight = float(np.sum(weights))
    if weight <= 0.0:
        raise ValueError("Merged component weight must be positive.")

    mean = np.sum(weights[:, None] * means, axis=0) / weight
    centered_means = means - mean
    variance = np.sum(
        weights[:, None] * (variances + centered_means ** 2),
        axis=0,
    ) / weight
    variance = np.clip(variance, 1e-10, None)
    strategy_values = np.asarray([
        float(component['strategy_value']) for component in components
    ])
    strategy_value = float(np.dot(weights, strategy_values) / weight)
    source_ids = sorted(
        {
            source_id
            for component in components
            for source_id in _source_agent_ids(component)
        },
        key=lambda value: str(value),
    )
    return {
        'weight': weight,
        'mean': mean.tolist(),
        'covariance': variance.tolist(),
        'strategy_value': strategy_value,
        'agent_id': 'merged',
        'source_agent_ids': source_ids,
    }


def diagonal_moment_match(distribution_a, distribution_b):
    """Backward-compatible two-component diagonal moment matching."""
    return diagonal_moment_match_cluster([distribution_a, distribution_b])


def full_moment_match_cluster(distributions):
    """Moment-match one final cluster for the full-covariance ablation."""
    components = list(distributions)
    if not components:
        raise ValueError("Cannot moment-match an empty distribution cluster.")

    validated = [_validate_distribution_shape(component, 'full') for component in components]
    means = np.stack([mean for mean, _ in validated])
    covariances = np.stack([covariance for _, covariance in validated])
    weights = np.asarray([float(component['weight']) for component in components])
    weight = float(np.sum(weights))
    if weight <= 0.0:
        raise ValueError("Merged component weight must be positive.")

    mean = np.sum(weights[:, None] * means, axis=0) / weight
    covariance = np.zeros_like(covariances[0], dtype=float)
    for component_weight, component_mean, component_covariance in zip(
        weights, means, covariances
    ):
        centered_mean = component_mean - mean
        covariance += component_weight * (
            component_covariance + np.outer(centered_mean, centered_mean)
        )
    covariance /= weight
    covariance = 0.5 * (covariance + covariance.T)
    strategy_values = np.asarray([
        float(component['strategy_value']) for component in components
    ])
    strategy_value = float(np.dot(weights, strategy_values) / weight)
    source_ids = sorted(
        {
            source_id
            for component in components
            for source_id in _source_agent_ids(component)
        },
        key=lambda value: str(value),
    )
    return {
        'weight': weight,
        'mean': mean.tolist(),
        'covariance': covariance.tolist(),
        'strategy_value': strategy_value,
        'agent_id': 'merged',
        'source_agent_ids': source_ids,
    }


def full_moment_match(distribution_a, distribution_b):
    """Backward-compatible two-component full-covariance moment matching."""
    return full_moment_match_cluster([distribution_a, distribution_b])


def _distribution_sort_key(distribution):
    covariance = np.asarray(distribution['covariance'], dtype=float).reshape(-1)
    return (
        tuple(np.asarray(distribution['mean'], dtype=float).tolist()),
        tuple(covariance.tolist()),
        float(distribution['weight']),
        float(distribution['strategy_value']),
        tuple(str(value) for value in _source_agent_ids(distribution)),
    )


def iterative_merge_distributions(
    distributions,
    threshold=RMS_EUCLIDEAN_THRESHOLD,
    covariance_mode='diag',
):
    """Cluster original means by complete linkage, then moment-match each cluster once."""
    if covariance_mode not in {'diag', 'full'}:
        raise ValueError("covariance_mode must be 'diag' or 'full'.")
    if threshold < 0.0:
        raise ValueError("RMS Euclidean distance threshold must be non-negative.")

    pool = copy.deepcopy(list(distributions))
    for distribution in pool:
        _validate_distribution_shape(distribution, covariance_mode)
        distribution['source_agent_ids'] = _source_agent_ids(distribution)
    # Canonical mean ordering makes geometric tie-breaking independent of upload order.
    pool.sort(key=lambda distribution: tuple(
        np.asarray(distribution['mean'], dtype=float).tolist()
    ))
    if len(pool) < 2:
        return pool

    # Freeze all distances between original component means. Covariances and
    # intermediate moment-matched components never influence cluster membership.
    means = [np.asarray(distribution['mean'], dtype=float) for distribution in pool]
    pairwise_distances = np.zeros((len(pool), len(pool)), dtype=float)
    for i in range(len(pool) - 1):
        for j in range(i + 1, len(pool)):
            distance = rms_euclidean_distance(means[i], means[j])
            pairwise_distances[i, j] = distance
            pairwise_distances[j, i] = distance

    clusters = [(index,) for index in range(len(pool))]
    tie_tolerance = 1e-12

    while len(clusters) >= 2:
        best_distance = float('inf')
        best_pair = None
        for i in range(len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                # Complete-linkage distance is the largest original-mean
                # distance across the two candidate clusters.
                distance = max(
                    pairwise_distances[left, right]
                    for left in clusters[i]
                    for right in clusters[j]
                )
                if distance < best_distance - tie_tolerance:
                    best_distance = distance
                    best_pair = (i, j)
        if best_pair is None or best_distance > threshold:
            break

        i_star, j_star = best_pair
        merged_cluster = tuple(sorted(clusters[i_star] + clusters[j_star]))
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in (i_star, j_star)
        ]
        clusters.append(merged_cluster)
        clusters.sort()

    moment_match_cluster = (
        diagonal_moment_match_cluster
        if covariance_mode == 'diag'
        else full_moment_match_cluster
    )
    merged_pool = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged_pool.append(copy.deepcopy(pool[cluster[0]]))
        else:
            merged_pool.append(moment_match_cluster([pool[index] for index in cluster]))
    merged_pool.sort(key=_distribution_sort_key)
    return merged_pool

def thompson_sampling_optimization(ts_model, bounds, num_samples=1):
    test_X = torch.rand(1000, bounds.shape[1], device=DEVICE, dtype=DTYPE) * (bounds[1] - bounds[0]) + bounds[0]
    with torch.no_grad():
        posterior = ts_model.posterior(test_X)
        sampler = SobolQMCNormalSampler(num_samples)
        sampled_funcs = sampler(posterior).squeeze(-1)
    candidates = []
    for i in range(num_samples):
        candidates.append(test_X[sampled_funcs[i].argmax()])
    return torch.stack(candidates)

# ================== Orthogonal feature sampling utilities ==================
def get_orf_features(orf_model, X):
    base_kernel = orf_model.covar_module.base_kernel
    ls = base_kernel.lengthscale
    if ls.shape[-1] == 1: 
        ls = ls.expand(-1, X.shape[-1])
        
    x_scaled = X / ls
    
    base_kernel._init_weights(X.shape[-1])
    proj = x_scaled @ base_kernel.random_weights.T + base_kernel.random_phases
    phi = torch.cos(proj) * math.sqrt(2.0 / base_kernel.num_samples)
    
    outputscale = orf_model.covar_module.outputscale.sqrt()
    return phi * outputscale

def sample_orf_weights(exact_model, orf_model):
    X_train = exact_model.train_inputs[0]
    Y_train = exact_model.train_targets.unsqueeze(-1)
    
    noise_tensor = exact_model.likelihood.noise
    noise_var = noise_tensor.mean().item() if noise_tensor.numel() > 1 else noise_tensor.item()

    Phi = get_orf_features(orf_model, X_train)
    M_dim = Phi.shape[-1]
    I = torch.eye(M_dim, device=X_train.device, dtype=X_train.dtype)
    
    S = Phi.T @ Phi + (noise_var + 1e-6) * I
    L = torch.linalg.cholesky(S)
    
    PhiY = Phi.T @ Y_train
    mu_w = torch.cholesky_solve(PhiY, L)
    
    z = torch.randn(M_dim, 1, device=X_train.device, dtype=X_train.dtype)
    v = torch.linalg.solve_triangular(L.T, z, upper=True)
    w_sample = mu_w + math.sqrt(noise_var) * v
    
    return w_sample.squeeze(-1), mu_w.squeeze(-1)

def sample_orf_weight_paths(exact_model, orf_model, num_paths):
    """Batch-sample ORF posterior weights for multiple Thompson paths."""
    X_train = exact_model.train_inputs[0]
    Y_train = exact_model.train_targets
    if Y_train.dim() == 1:
        Y_train = Y_train.unsqueeze(-1)
    
    noise_tensor = exact_model.likelihood.noise
    noise_var = noise_tensor.mean().item() if noise_tensor.numel() > 1 else noise_tensor.item()

    Phi = get_orf_features(orf_model, X_train)
    feature_dim = Phi.shape[-1]
    I = torch.eye(feature_dim, device=X_train.device, dtype=X_train.dtype)
    
    S = Phi.T @ Phi + (noise_var + 1e-6) * I
    L = torch.linalg.cholesky(S)
    
    PhiY = Phi.T @ Y_train
    mu_w = torch.cholesky_solve(PhiY, L).squeeze(-1)
    
    z = torch.randn(feature_dim, num_paths, device=X_train.device, dtype=X_train.dtype)
    v = torch.linalg.solve_triangular(L.T, z, upper=True)
    w_samples = mu_w.unsqueeze(-1) + math.sqrt(noise_var) * v
    
    return w_samples.T, mu_w

def thompson_sampling_optimization_orf(exact_model, orf_model, bounds, num_samples=1):
    """Generate near-optimal candidates from ORF Thompson sample paths."""
    test_X = torch.rand(1000, bounds.shape[1], device=DEVICE, dtype=DTYPE) * (bounds[1] - bounds[0]) + bounds[0]
    
    with torch.no_grad():
        Phi = get_orf_features(orf_model, test_X)
        w_samples, mu_w = sample_orf_weight_paths(exact_model, orf_model, num_samples)
        
        exact_mu = exact_model.posterior(test_X).mean.squeeze(-1)
        mean_orf = Phi @ mu_w
        sampled_funcs = exact_mu.unsqueeze(-1) + (Phi @ w_samples.T - mean_orf.unsqueeze(-1))
    
    best_indices = torch.argmax(sampled_funcs, dim=0)
    return test_X[best_indices]

def _torch_cholesky_with_jitter(covariance):
    """Cholesky factorization for full guidance covariances without inversion."""
    global FULL_COVARIANCE_JITTER_EVENTS
    covariance = 0.5 * (covariance + covariance.transpose(-2, -1))
    identity = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
    for jitter in (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        factor, info = torch.linalg.cholesky_ex(covariance + jitter * identity)
        if torch.all(info == 0):
            if jitter > 0.0:
                FULL_COVARIANCE_JITTER_EVENTS += 1
            return factor
    raise ValueError("Full guidance covariance is not positive definite after jitter up to 1e-3.")


def _prepare_guidance_params(received_distributions, covariance_mode):
    if covariance_mode not in {'diag', 'full'}:
        raise ValueError("covariance_mode must be 'diag' or 'full'.")
    if not received_distributions:
        return None
    weights = torch.tensor(
        [distribution['reset_weight'] for distribution in received_distributions],
        device=DEVICE,
        dtype=DTYPE,
    )
    means = torch.tensor(
        [distribution['mean'] for distribution in received_distributions],
        device=DEVICE,
        dtype=DTYPE,
    )
    covariance_values = []
    for distribution in received_distributions:
        mean, covariance = _validate_distribution_shape(distribution, covariance_mode)
        if mean.size != DIM:
            raise ValueError(f"Guidance component dimension must equal DIM={DIM}, got {mean.size}.")
        covariance_tensor = torch.tensor(covariance, device=DEVICE, dtype=DTYPE)
        if covariance_mode == 'diag':
            covariance_values.append(torch.clamp(covariance_tensor, min=1e-10))
        else:
            covariance_values.append(_torch_cholesky_with_jitter(covariance_tensor))
    return weights, means, torch.stack(covariance_values)


def _guidance_field(X, guidance_params, covariance_mode):
    """Evaluate the packet-defined spatial guidance field G(x)."""
    if guidance_params is None:
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    weights, means, covariance_values = guidance_params
    G_values = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
    for index in range(len(weights)):
        delta = X - means[index]
        if covariance_mode == 'diag':
            dist_sq = torch.sum(
                delta ** 2 / (covariance_values[index] + 1e-10), dim=-1
            )
        else:
            solved = torch.cholesky_solve(
                delta.unsqueeze(-1), covariance_values[index]
            ).squeeze(-1)
            dist_sq = torch.sum(delta * solved, dim=-1)
        G_values = G_values + weights[index] * torch.exp(-0.5 * dist_sq)
    return G_values


def _domain_average_guidance(guidance_params, covariance_mode, num_points):
    """Estimate domain-average G with a deterministic unscrambled Sobol design."""
    if num_points <= 0:
        raise ValueError("GUIDANCE_AVERAGE_NUM_POINTS must be positive.")
    if guidance_params is None:
        return torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    sobol = torch.quasirandom.SobolEngine(dimension=DIM, scramble=False)
    unit_points = sobol.draw(num_points).to(device=DEVICE, dtype=DTYPE)
    probe_points = bounds[0] + unit_points * (bounds[1] - bounds[0])
    return _guidance_field(
        probe_points, guidance_params, covariance_mode
    ).mean()


def _guidance_scaling_factor(
    X,
    guidance_params,
    covariance_mode,
    lambda_guidance,
    X_observed=None,
    uniform_guidance_value=None,
):
    """Compute S(x)=1+lambda*G(x) for diagonal or full Gaussian components."""
    if uniform_guidance_value is None:
        G_values = _guidance_field(X, guidance_params, covariance_mode)
    else:
        # The uniform ablation applies the same domain-average field everywhere.
        G_values = torch.ones(
            X.shape[:-1], device=X.device, dtype=X.dtype
        ) * uniform_guidance_value.to(device=X.device, dtype=X.dtype)
    scaling = 1.0 + lambda_guidance * G_values
    if X_observed is not None and uniform_guidance_value is None:
        min_distance = torch.min(torch.cdist(X, X_observed), dim=-1).values
        scaling = torch.where(min_distance < 1e-5, torch.ones_like(scaling), scaling)
    return scaling


class GuidanceScaledKernel(Kernel):
    """Non-stationary kernel k_tilde(x,x') = S(x) k(x,x') S(x')."""

    def __init__(
        self,
        base_kernel,
        received_distributions,
        lambda_guidance,
        X_observed=None,
        covariance_mode='diag',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.lambda_guidance = lambda_guidance
        self.X_observed = X_observed
        self.covariance_mode = covariance_mode
        self.guidance_params = _prepare_guidance_params(
            received_distributions, covariance_mode
        )

    def _get_scaling_factor(self, X):
        return _guidance_scaling_factor(
            X,
            self.guidance_params,
            self.covariance_mode,
            self.lambda_guidance,
            self.X_observed,
        )

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        base_covar = self.base_kernel(
            x1,
            x2,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )
        if hasattr(base_covar, 'to_dense'):
            base_covar = base_covar.to_dense()
        scale_1 = self._get_scaling_factor(x1)
        scale_2 = self._get_scaling_factor(x2)
        if diag:
            return base_covar * scale_1 * scale_2
        return base_covar * scale_1.unsqueeze(-1) * scale_2.unsqueeze(-2)


# ================== Federated interventional GP model ==================
class FederatedInterventionalGP(Model):
    """
    FI-GP (Federated Interventional GP):
    Scales the posterior covariance by S * \Sigma * S
    to incorporate spatial guidance G(x) without changing the posterior mean.
    """
    def __init__(
        self,
        base_model,
        orf_model,
        received_distributions,
        lambda_guidance,
        X_observed=None,
        enable_sample_path=False,
        covariance_mode='diag',
    ):
        super().__init__()
        self.base_model = base_model
        self.orf_model = orf_model
        self.lambda_guidance = lambda_guidance
        self.X_observed = X_observed
        self.enable_sample_path = enable_sample_path
        self.covariance_mode = covariance_mode
        self.gmm_params = _prepare_guidance_params(
            received_distributions, covariance_mode
        )
        self.uniform_guidance_value = None
        if not SPATIAL_GUIDANCE:
            self.uniform_guidance_value = _domain_average_guidance(
                self.gmm_params,
                covariance_mode,
                GUIDANCE_AVERAGE_NUM_POINTS,
            )

        # Sample a continuous feature residual path for Thompson sampling.
        if self.enable_sample_path and orf_model is not None:
            self.w_sample, self.mu_w = sample_orf_weights(base_model, orf_model)

    @property
    def num_outputs(self) -> int:
        return self.base_model.num_outputs

    def _get_scaling_factor(self, X):
        """Compute the scaling factor S(x) = 1 + \lambda G(x)."""
        return _guidance_scaling_factor(
            X,
            self.gmm_params,
            self.covariance_mode,
            self.lambda_guidance,
            self.X_observed,
            self.uniform_guidance_value,
        )

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        """Posterior interface for analytic acquisitions such as UCB and NEI."""
        real_post = self.base_model.posterior(X, observation_noise=observation_noise)
        real_mvn = real_post.distribution
        
        # Broadcast S * Sigma * S without allocating a diagonal matrix.
        S_diag = self._get_scaling_factor(X)
        virtual_covar = (
            real_mvn.covariance_matrix
            * S_diag.unsqueeze(-1)
            * S_diag.unsqueeze(-2)
        )
        
        try:
            virtual_mvn = MultivariateNormal(real_mvn.mean, virtual_covar)
            
        except RuntimeError: 
            from linear_operator.operators import DenseLinearOperator
            
            jitter_val = 1e-4  
            jitter_mat = torch.eye(
                virtual_covar.shape[-1],
                device=virtual_covar.device,
                dtype=virtual_covar.dtype,
            ) * jitter_val
            virtual_covar_safe = DenseLinearOperator(virtual_covar + jitter_mat)
            virtual_mvn = MultivariateNormal(real_mvn.mean, virtual_covar_safe)
            
        return GPyTorchPosterior(virtual_mvn)

    def sample_path(self):
        """Differentiable path interface for continuous Thompson sampling."""
        if self.orf_model is None:
            raise RuntimeError("ORF model is required for TS sample_path().")
        if not hasattr(self, 'w_sample') or not hasattr(self, 'mu_w'):
            self.w_sample, self.mu_w = sample_orf_weights(self.base_model, self.orf_model)
        
        base_model = self.base_model
        orf_model = self.orf_model
        w_sample = self.w_sample
        mu_w = self.mu_w
        get_S = self._get_scaling_factor

        class VirtualSampledFunction(torch.nn.Module):
            def forward(self, X):
                exact_mu = base_model.posterior(X).mean.squeeze(-1)
                
                Phi = get_orf_features(orf_model, X)
                f_path = (Phi @ w_sample.unsqueeze(-1)).squeeze(-1)
                mu_orf = (Phi @ mu_w.unsqueeze(-1)).squeeze(-1)
                epsilon = f_path - mu_orf 
                
                S_x = get_S(X)
                
                # Apply spatially varying scaling to the continuous sample path.
                return exact_mu + epsilon * S_x

        return VirtualSampledFunction()

# ================== Continuous Thompson sampling acquisition ==================
class ContinuousThompsonSampling(AcquisitionFunction):
    def __init__(self, model):
        super().__init__(model)
        # Request one continuous sample path from the FI-GP at initialization.
        self.sampled_path = model.sample_path()
        
    def forward(self, X):
        # Evaluate the sampled path with continuous gradients.
        return self.sampled_path(X).squeeze(-1)


class DirectModelContinuousThompsonSampling(AcquisitionFunction):
    """Continuous ORF Thompson path drawn directly from an intervened local GP."""

    def __init__(self, model, orf_model):
        super().__init__(model)
        self.orf_model = orf_model
        self.w_sample, self.mu_w = sample_orf_weights(model, orf_model)

    def forward(self, X):
        exact_mu = self.model.posterior(X).mean.squeeze(-1)
        features = get_orf_features(self.orf_model, X)
        sampled = (features @ self.w_sample.unsqueeze(-1)).squeeze(-1)
        mean_orf = (features @ self.mu_w.unsqueeze(-1)).squeeze(-1)
        return (exact_mu + sampled - mean_orf).squeeze(-1)
