
import jax
import jax.numpy as jnp

from .linear_layer import LinearMask
from .. import functional
from ..functional import StepDoubleGaussianGrad, quantize_tensor
from flax import linen as nn
from flax.linen.initializers import zeros, normal, xavier_uniform

################################################################
# Neuron update functional
################################################################

# default values for time constants
DEFAULT_ALIF_TAU_M = 20.
DEFAULT_ALIF_TAU_ADP = 20.

# base threshold
DEFAULT_ALIF_THETA = 0.01

DEFAULT_ALIF_BETA = 1.8

@jax.jit
def alif_update(x, z, u, a, alpha, rho, beta=DEFAULT_ALIF_BETA, theta=DEFAULT_ALIF_THETA):
    a = a * rho + z * (1.0 - rho)              # adapt spike accumulator
    theta_t = theta + a * beta                  # dynamic threshold
    u = u * alpha + x * (1.0 - alpha)          # membrane potential update
    z = StepDoubleGaussianGrad(u - theta_t) # generate spike
    u = u - z * theta_t                         # soft reset membrane potential
    return z, u, a

class ALIFCell(nn.Module):
    input_size: int
    layer_size: int

    adaptive_tau_mem_mean: float
    adaptive_tau_mem_std: float
    adaptive_tau_adp_mean: float
    adaptive_tau_adp_std: float

    tau_mem: float = DEFAULT_ALIF_TAU_M
    adaptive_tau_mem: bool = True

    tau_adp: float = DEFAULT_ALIF_TAU_ADP
    adaptive_tau_adp: bool = True

    bias: bool = False
    mask_prob: float = 0.
    pruning: bool = False

    def setup(self):
        if self.pruning:
            self.linear = LinearMask(
                in_features=self.input_size,
                out_features=self.layer_size,
                bias=self.bias,
                mask_prob=self.mask_prob,
                lbd=self.input_size - self.layer_size,
                ubd=self.input_size
            )
        else:
            self.linear = nn.Dense(
                self.layer_size,
                use_bias=self.bias,
                kernel_init=xavier_uniform(),
                bias_init=zeros
            )

        # Initialize tau_mem
        tau_mem_init = self.tau_mem * jnp.ones(self.layer_size)
        if self.adaptive_tau_mem:
            self.tau_mem_param = self.param(
                "tau_mem",
                lambda key, shape: normal(stddev=self.adaptive_tau_mem_std)(key, shape) + self.adaptive_tau_mem_mean,
                (self.layer_size,)
            )
        else:
            self.tau_mem_param = self.variable("constants", "tau_mem", lambda: tau_mem_init)

        # Initialize tau_adp
        tau_adp_init = self.tau_adp * jnp.ones(self.layer_size)
        if self.adaptive_tau_adp:
            self.tau_adp_param = self.param(
                "tau_adp",
                lambda key, shape: normal(stddev=self.adaptive_tau_adp_std)(key, shape) + self.adaptive_tau_adp_mean,
                (self.layer_size,)
            )
        else:
            self.tau_adp_param = self.variable("constants", "tau_adp", lambda: tau_adp_init)

    def __call__(self, x, state):
        z, u, a = state

        in_sum = self.linear(x)

        tau_mem = jnp.abs(self.tau_mem_param.value if isinstance(self.tau_mem_param, nn.Variable) else self.tau_mem_param)
        alpha = jnp.exp(-1.0 / tau_mem)

        tau_adp = jnp.abs(self.tau_adp_param.value if isinstance(self.tau_adp_param, nn.Variable) else self.tau_adp_param)
        rho = jnp.exp(-1.0 / tau_adp)

        z, u, a = alif_update(
            x=in_sum,
            z=z,
            u=u,
            a=a,
            alpha=alpha,
            rho=rho,
        )
        return z, u, a


# --- ALIFCellBP (with quantization) ---

class ALIFCellBP(ALIFCell):
    bit_precision: int = 32

    def __call__(self, x, state):
        z, u, a = state

        in_sum = self.linear(x)

        tau_mem = jnp.abs(self.tau_mem_param.value if isinstance(self.tau_mem_param, nn.Variable) else self.tau_mem_param)
        alpha = jnp.exp(-1.0 / tau_mem)
        alpha = quantize_tensor(alpha, self.bit_precision)

        tau_adp = jnp.abs(self.tau_adp_param.value if isinstance(self.tau_adp_param, nn.Variable) else self.tau_adp_param)
        rho = jnp.exp(-1.0 / tau_adp)
        rho = quantize_tensor(rho, self.bit_precision)

        z, u, a = alif_update(
            x=in_sum,
            z=z,
            u=u,
            a=a,
            alpha=alpha,
            rho=rho,
        )
        return z, u, a
