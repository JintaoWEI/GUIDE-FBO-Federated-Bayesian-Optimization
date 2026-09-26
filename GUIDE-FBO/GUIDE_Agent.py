"""
Agent module
Defines a federated agent that runs local Bayesian optimization.
"""
import torch
import numpy as np
import math
from torch import Tensor
from botorch.models.gp_regression import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import UpperConfidenceBound, qNoisyExpectedImprovement

from GUIDE_Config import *
from GUIDE_Orf import create_orf_model_from_exact_gp
from GUIDE_Utils import (truth, bounds, thompson_sampling_optimization,
                   thompson_sampling_optimization_orf, fit_dpgmm,
                   fit_single_gaussian, get_beta_value,
                   FederatedInterventionalGP, ContinuousThompsonSampling,
                   DirectModelContinuousThompsonSampling, GuidanceScaledKernel)


# Local optimization agent for GUIDE_Agent.
class Agent:
    def __init__(self, agent_id, cube_lb, cube_length, exp_id=0):
        self.agent_id = agent_id
        self.cube_lb = cube_lb.to(DEVICE)
        self.cube_length = cube_length.to(DEVICE)
        self.iteration = 0
        
        D = bounds.shape[1]
        
        current_rng_state = torch.get_rng_state()
        unique_heter_seed = SEED_BASE + exp_id * 1000 + agent_id
        torch.manual_seed(unique_heter_seed)

        domain_diagonal = torch.linalg.norm(bounds[1] - bounds[0])
        self.shift_z = torch.randn(1, D, device=DEVICE, dtype=DTYPE) * (HETER_SHIFT * domain_diagonal / math.sqrt(D))

        if HETER_ROTATION > 0:
            random_M = torch.randn(D, D, device=DEVICE, dtype=DTYPE)
            skew_symmetric = random_M - random_M.T
            self.rotation_R = torch.matrix_exp(HETER_ROTATION * skew_symmetric)
        else:
            self.rotation_R = torch.eye(D, device=DEVICE, dtype=DTYPE)

        torch.set_rng_state(current_rng_state)

        self.local_x = None
        self.local_y = None
        self.model = None
        self.orf_model = None  

        self.current_distributions = []
        self.received_distributions = []
        self.current_lambda = 0.0
        self.last_posterior_mean_difference = 0.0
        self.last_posterior_variance_difference = 0.0
        
        self.btv_value = -float('inf')  
        self.instantaneous_value = 0.0  

    def get_current_beta(self):
        return get_beta_value()

    def observation(self, X):
        shifted_X = X - self.shift_z
        transformed_X = torch.matmul(shifted_X, self.rotation_R.T)
        
        exact_y = truth(transformed_X) 
        
        if exact_y.dim() == 0: 
            exact_y = exact_y.unsqueeze(0).unsqueeze(1)
        elif exact_y.dim() == 1: 
            exact_y = exact_y.unsqueeze(-1)
        
        val = (exact_y[-1] * NORMALIZE_Y).item()
        self.instantaneous_value = val
        self.btv_value = max(self.btv_value, val)
        
        return exact_y + NOISE_SE * torch.randn_like(exact_y)

    def generate_initial_data(self, exp_id=0):
        current_rng_state = torch.get_rng_state()
        init_data_seed = SEED_BASE + exp_id * 1000 + self.agent_id + 5000 
        torch.manual_seed(init_data_seed)

        self.local_x = torch.rand(INITIAL_SAMPLES, bounds.shape[1], device=DEVICE, dtype=DTYPE) * self.cube_length + self.cube_lb
        self.local_y = self.observation(self.local_x)
        
        with torch.no_grad():
            shifted_X = self.local_x - self.shift_z
            transformed_X = torch.matmul(shifted_X, self.rotation_R.T)
            all_true_y = truth(transformed_X) * NORMALIZE_Y
            self.btv_value = all_true_y.max().item()
            
        self.update_model()
        torch.set_rng_state(current_rng_state)

    def update_model(self):
        noise_variance = (NOISE_SE ** 2) * torch.ones_like(self.local_y)
        self.model = FixedNoiseGP(self.local_x, self.local_y, noise_variance).to(DEVICE)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(DEVICE)
        fit_gpytorch_model(mll)

        if (
            DIRECT_SURROGATE_INTERVENTION
            and self.received_distributions
            and self.current_lambda > 0.0
        ):
            # Compare the directly intervened surrogate with the fitted standard local GP.
            probe_points = torch.stack([
                bounds[0] + 0.25 * (bounds[1] - bounds[0]),
                bounds[0] + 0.50 * (bounds[1] - bounds[0]),
                bounds[0] + 0.75 * (bounds[1] - bounds[0]),
            ])
            with torch.no_grad():
                baseline_posterior = self.model.posterior(probe_points)
                baseline_mean = baseline_posterior.mean.detach().clone()
                baseline_variance = baseline_posterior.variance.detach().clone()

            self.model.covar_module = GuidanceScaledKernel(
                base_kernel=self.model.covar_module,
                received_distributions=self.received_distributions,
                lambda_guidance=self.current_lambda,
                X_observed=self.local_x,
                covariance_mode=DPGMM_COVARIANCE_TYPE,
            ).to(DEVICE)
            intervened_mll = ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            ).to(DEVICE)
            fit_gpytorch_model(intervened_mll)
            with torch.no_grad():
                intervened_posterior = self.model.posterior(probe_points)
                self.last_posterior_mean_difference = torch.mean(torch.abs(
                    intervened_posterior.mean - baseline_mean
                )).item()
                self.last_posterior_variance_difference = torch.mean(torch.abs(
                    intervened_posterior.variance - baseline_variance
                )).item()
        else:
            self.last_posterior_mean_difference = 0.0
            self.last_posterior_variance_difference = 0.0

    def fit_gmm_and_extract_distributions(self):
        needs_orf_model = USE_ORF_FOR_TS or ACQ_FUNCTION == 'TS'
        if needs_orf_model:
            self.orf_model = create_orf_model_from_exact_gp(self.model, num_orf_samples=ORF_NUM_SAMPLES)
        else:
            self.orf_model = None
            
        if USE_ORF_FOR_TS:
            ts_candidates = thompson_sampling_optimization_orf(self.model, self.orf_model, bounds, M)
        else:
            ts_candidates = thompson_sampling_optimization(self.model, bounds, M)
        if BELIEF_EXTRACTION_MODE == 'dpgmm':
            gmm_result = fit_dpgmm(ts_candidates)
        elif BELIEF_EXTRACTION_MODE == 'single_gaussian':
            if DPGMM_COVARIANCE_TYPE != 'diag':
                raise ValueError(
                    "single_gaussian belief extraction requires diagonal covariance."
                )
            gmm_result = fit_single_gaussian(ts_candidates)
        else:
            raise ValueError(
                "BELIEF_EXTRACTION_MODE must be 'dpgmm' or 'single_gaussian'."
            )
        
        self.current_distributions = []
        weights = gmm_result['weights']

        if len(weights) > 0:
            best_idx = int(np.argmax(weights))
            best_mean = gmm_result['means'][best_idx]
            
            mean_tensor = torch.tensor(best_mean, device=DEVICE, dtype=DTYPE).unsqueeze(0)
            with torch.no_grad():
                post = self.model.posterior(mean_tensor)
                lcb = post.mean.item() - 1.0 * post.variance.sqrt().item()
            
            local_y_mean = self.local_y.mean().item()
            local_y_std = self.local_y.std().item() + 1e-6
            z_score_lcb = (lcb - local_y_mean) / local_y_std

            self.current_distributions.append({
                'weight': weights[best_idx], 
                'mean': best_mean, 
                'covariance': gmm_result['covariances'][best_idx], 
                'strategy_value': z_score_lcb, 
                'agent_id': self.agent_id,
                'source_agent_ids': [self.agent_id],
            })
            
    def receive_distributions_from_server(self, dists):
        expected_shape = (DIM,) if DPGMM_COVARIANCE_TYPE == 'diag' else (DIM, DIM)
        for distribution in dists:
            covariance = np.asarray(distribution['covariance'], dtype=float)
            if covariance.shape != expected_shape:
                raise ValueError(
                    f"Agent {self.agent_id} expected {DPGMM_COVARIANCE_TYPE} "
                    f"covariance shape {expected_shape}, got {covariance.shape}."
                )
        self.received_distributions = dists

    def _calculate_adaptive_lambda(self):
        t = max(1, self.iteration) 
        lambda_t = LAMBDA_MAX / math.sqrt(t)
        return lambda_t

    def select_next_observation_point(self):
        self.iteration += 1
        current_lambda = self._calculate_adaptive_lambda()
        self.current_lambda = current_lambda
        
        num_restarts = ACQ_NUM_RESTARTS
        raw_samples = ACQ_RAW_SAMPLES
        batch_limit = ACQ_BATCH_LIMIT
        max_itr = ACQ_MAX_ITER

        # Direct-surrogate ablation modifies and refits the local GP itself.
        if DIRECT_SURROGATE_INTERVENTION:
            self.update_model()
            decision_model = self.model
            if ACQ_FUNCTION == 'TS':
                self.orf_model = create_orf_model_from_exact_gp(
                    self.model, num_orf_samples=ORF_NUM_SAMPLES
                )
        else:
            # Default GUIDE uses a decision-time FI-GP wrapper.
            decision_model = FederatedInterventionalGP(
                base_model=self.model,
                orf_model=self.orf_model,
                received_distributions=self.received_distributions,
                lambda_guidance=current_lambda,
                X_observed=self.local_x,
                enable_sample_path=(ACQ_FUNCTION == 'TS'),
                covariance_mode=DPGMM_COVARIANCE_TYPE,
            )

        # 2. Use the native acquisition function for the local decision step.
        if ACQ_FUNCTION == 'UCB':
            actual_beta = self.get_current_beta() ** 2 
            acq = UpperConfidenceBound(model=decision_model, beta=actual_beta)
        elif ACQ_FUNCTION in ['NEI', 'EI']: 
            acq = qNoisyExpectedImprovement(
                model=decision_model,
                X_baseline=self.local_x,
            )
        elif ACQ_FUNCTION == 'TS':
            if DIRECT_SURROGATE_INTERVENTION:
                acq = DirectModelContinuousThompsonSampling(
                    model=decision_model, orf_model=self.orf_model
                )
            else:
                acq = ContinuousThompsonSampling(model=decision_model)
            
        candidates, _ = optimize_acqf(
            acq_function=acq, bounds=bounds, q=1, num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": batch_limit, "maxiter": max_itr},
        )
        
        self.selected_point = candidates
        self.selected_value = self.observation(self.selected_point)
        
        self.local_x = torch.cat([self.local_x, self.selected_point])
        self.local_y = torch.cat([self.local_y, self.selected_value])
        
        self.update_model()

    def get_btv_value(self): return self.btv_value
    def get_instantaneous_value(self): return self.instantaneous_value
