import torch
import math
from gpytorch.kernels import Kernel
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import GreaterThan
from botorch.models.gpytorch import GPyTorchModel
from linear_operator.operators import DenseLinearOperator

from GUIDE_Config import DEVICE, DTYPE, NOISE_SE, ORF_NUM_SAMPLES


class Matern25ORFKernel(Kernel):
    """
    ORF Approximation of Matérn-2.5 Kernel
    """
    
    has_lengthscale = True
    
    def __init__(self, num_samples, ard_num_dims=None, use_orf=True, **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        self.num_samples = num_samples
        self.use_orf = use_orf
        self.student_t_df = 5.0
        self._weights_initialized = False
        self.register_buffer('random_weights', None)
        self.register_buffer('random_phases', None)
    
    def _init_weights(self, num_dims):
        if self._weights_initialized and self.random_weights is not None:
            if self.random_weights.shape[1] == num_dims: return
        
        device = self.lengthscale.device
        dtype = self.lengthscale.dtype
        
        if self.use_orf: weights = self._sample_orthogonal_student_t(num_dims, device, dtype)
        else: weights = self._sample_student_t(self.num_samples, num_dims, device, dtype)
        
        self.register_buffer('random_weights', weights)
        self.register_buffer('random_phases', torch.rand(self.num_samples, device=device, dtype=dtype) * 2 * math.pi)
        self._weights_initialized = True
    
    def _sample_student_t(self, n_samples, n_dims, device, dtype):
        df = self.student_t_df
        z = torch.randn(n_samples, n_dims, device=device, dtype=dtype)
        chi2 = 2.0 * torch._standard_gamma(torch.full((n_samples, 1), df / 2.0, device=device, dtype=dtype))
        return z / torch.sqrt(chi2 / df)
    
    def _sample_orthogonal_student_t(self, num_dims, device, dtype):
        S, D, df = self.num_samples, num_dims, self.student_t_df
        weights_list = []
        nb_full_blocks = S // D
        
        for _ in range(nb_full_blocks):
            G = torch.randn(D, D, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(G)
            weights_list.append(Q)
        
        remaining = S - nb_full_blocks * D
        if remaining > 0:
            G = torch.randn(D, D, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(G)
            weights_list.append(Q[:remaining])
        
        orthogonal_matrix = torch.cat(weights_list, dim=0)
        chi2_samples = 2.0 * torch._standard_gamma(torch.full((S, 1), df / 2.0, device=device, dtype=dtype))
        chi_D = 2.0 * torch._standard_gamma(torch.full((S, 1), D / 2.0, device=device, dtype=dtype))
        gaussian_norms = torch.sqrt(chi_D)
        student_t_scale = torch.sqrt(df / chi2_samples)
        
        return orthogonal_matrix * gaussian_norms * student_t_scale
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        num_dims = x1.shape[-1]
        self._init_weights(num_dims)
        
        ls = self.lengthscale
        if ls.shape[-1] == 1: ls = ls.expand(-1, num_dims)
        
        x1_scaled = x1 / ls
        x2_scaled = x2 / ls
        
        proj1 = x1_scaled @ self.random_weights.T + self.random_phases
        proj2 = x2_scaled @ self.random_weights.T + self.random_phases
        
        phi1 = torch.cos(proj1) * math.sqrt(2.0 / self.num_samples)
        phi2 = torch.cos(proj2) * math.sqrt(2.0 / self.num_samples)
        
        if diag: return (phi1 * phi2).sum(dim=-1)
        else: return phi1 @ phi2.transpose(-2, -1)


class Matern25ORFGPModel(ExactGP, GPyTorchModel):
    _num_outputs = 1
    
    def __init__(self, train_x, train_y, likelihood, num_orf_samples, ard_num_dims=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        from gpytorch.kernels import ScaleKernel
        base_kernel = Matern25ORFKernel(num_samples=num_orf_samples, ard_num_dims=ard_num_dims, use_orf=True)
        self.covar_module = ScaleKernel(base_kernel)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if isinstance(covar_x, torch.Tensor):
            jitter = 1e-6 * torch.eye(covar_x.shape[-1], device=covar_x.device, dtype=covar_x.dtype)
            if covar_x.dim() == 3: jitter = jitter.unsqueeze(0)
            covar_x = DenseLinearOperator(covar_x + jitter)
        return MultivariateNormal(mean_x, covar_x)


def create_orf_model_from_exact_gp(exact_gp_model, num_orf_samples=ORF_NUM_SAMPLES):
    train_x = exact_gp_model.train_inputs[0].detach().clone().to(device=DEVICE, dtype=DTYPE)
    train_y = exact_gp_model.train_targets.detach().clone().to(device=DEVICE, dtype=DTYPE)
    if train_y.dim() == 2 and train_y.shape[-1] == 1: train_y = train_y.squeeze(-1)
    
    noise_var = NOISE_SE ** 2
    noise_constraint = GreaterThan(noise_var * 0.1)
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
    likelihood.noise = noise_var
    likelihood.noise_covar.raw_noise.requires_grad_(False)
    likelihood = likelihood.to(device=DEVICE, dtype=DTYPE)
    
    # Direct-surrogate GUIDE wraps the fitted ScaleKernel in GuidanceScaledKernel.
    # Unwrap only for copying the original stationary kernel hyperparameters;
    # the ORF feature construction itself remains unchanged.
    source_covar = exact_gp_model.covar_module
    while (
        hasattr(source_covar, 'base_kernel')
        and not hasattr(source_covar, 'outputscale')
    ):
        source_covar = source_covar.base_kernel

    ard_num_dims, source_ls, source_outputscale = None, None, None
    if hasattr(source_covar, 'outputscale'):
        source_outputscale = source_covar.outputscale.detach().clone()
    if hasattr(source_covar, 'base_kernel'):
        base_kernel = source_covar.base_kernel
        lengthscale = getattr(base_kernel, 'lengthscale', None)
        if lengthscale is not None:
            source_ls = lengthscale.detach().clone()
            if source_ls.numel() > 1: ard_num_dims = source_ls.numel()
    
    orf_model = Matern25ORFGPModel(train_x, train_y, likelihood, num_orf_samples, ard_num_dims).to(device=DEVICE, dtype=DTYPE)
    _copy_hyperparameters(
        exact_gp_model, orf_model, source_ls, source_outputscale
    )
    orf_model.eval()
    likelihood.eval()
    
    return orf_model

def _copy_hyperparameters(
    source_model, target_model, source_ls=None, source_outputscale=None
):
    with torch.no_grad():
        if source_outputscale is not None:
            target_model.covar_module.outputscale = source_outputscale.to(
                device=DEVICE, dtype=DTYPE
            )
        if source_ls is not None:
            target_model.covar_module.base_kernel.lengthscale = source_ls.to(device=DEVICE, dtype=DTYPE)
        if hasattr(source_model.mean_module, 'constant') and hasattr(target_model.mean_module, 'constant'):
            target_model.mean_module.constant.data.copy_(source_model.mean_module.constant.detach().clone().to(device=DEVICE, dtype=DTYPE))
