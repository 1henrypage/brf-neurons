import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import normal, zeros
from typing import Tuple

from jax._src.nn.initializers import xavier_uniform

from jax_snn.functional import StepLinearGrad, quantize_tensor

DEFAULT_LI_TAU_M = 20.
DEFAULT_LI_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_LI_ADAPTIVE_TAU_M_STD = 5.

DEFAULT_LIF_TAU_M = 20.
DEFAULT_LIF_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_LIF_ADAPTIVE_TAU_M_STD = 5.


@jax.jit
def li_update(x: jnp.ndarray, u: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    return u * alpha + x * (1.0 - alpha)


@jax.jit
def lif_update(x: jnp.ndarray, u: jnp.ndarray, alpha: jnp.ndarray, theta: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    u = u * alpha + x * (1.0 - alpha)
    z = StepLinearGrad(u - theta)
    u = u - z * theta
    return z, u


class LICell(nn.Module):
    input_size: int
    layer_size: int
    tau_mem: float = DEFAULT_LI_TAU_M
    adaptive_tau_mem: bool = True
    adaptive_tau_mem_mean: float = DEFAULT_LI_ADAPTIVE_TAU_M_MEAN
    adaptive_tau_mem_std: float = DEFAULT_LI_ADAPTIVE_TAU_M_STD
    bias: bool = False

    def setup(self):
        self.linear = nn.Dense(
            features=self.layer_size,
            use_bias=self.bias,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
        )

        if self.adaptive_tau_mem:
            self.tau_mem_param = self.param(
                "tau_mem",
                lambda key, shape: normal(stddev=self.adaptive_tau_mem_std)(key, shape) + self.adaptive_tau_mem_mean,
                (self.layer_size,),
            )
        else:
            self.tau_mem_param = jnp.full((self.layer_size,), self.tau_mem)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        in_sum = self.linear(x)

        # Access the parameter created in setup
        tau_mem = jnp.abs(self.tau_mem_param)

        alpha = jnp.exp(-1.0 / tau_mem)

        u = li_update(in_sum, u, alpha)
        return u


class LIFCell(nn.Module):
    input_size: int
    layer_size: int
    tau_mem: float = DEFAULT_LIF_TAU_M
    adaptive_tau_mem: bool = True
    adaptive_tau_mem_mean: float = DEFAULT_LIF_ADAPTIVE_TAU_M_MEAN
    adaptive_tau_mem_std: float = DEFAULT_LIF_ADAPTIVE_TAU_M_STD
    bias: bool = False

    def setup(self):
        self.linear = nn.Dense(
            features=self.layer_size,
            use_bias=self.bias,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
        )

        if self.adaptive_tau_mem:
            self.tau_mem_param = self.param(
                "tau_mem",
                lambda key, shape: normal(stddev=self.adaptive_tau_mem_std)(key, shape) + self.adaptive_tau_mem_mean,
                (self.layer_size,),
            )
        else:
            self.tau_mem_param = jnp.full((self.layer_size,), self.tau_mem)

    def __call__(self, x: jnp.ndarray, state: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z, u = state

        in_sum = self.linear(x)

        # Access the parameter created in setup
        tau_mem = jnp.abs(self.tau_mem_param)

        alpha = jnp.sigmoid(-1.0 / jnp.abs(tau_mem))

        z, u = lif_update(x=in_sum, u=u, alpha=alpha)
        return z, u


class LICellSigmoid(LICell):
    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        # Reusing `self.linear` from parent `LICell`'s setup
        in_sum = self.linear(x)

        # Reusing `self.tau_mem_param` from parent `LICell`'s setup
        # The original code re-defined it. To properly convert, it needs to use the parent's.
        # If this param is *different* from the parent's, LICellSigmoid needs its own setup.
        # Assuming for now it uses the one defined by the parent, but its *calculation* of alpha differs.
        tau_mem = self.tau_mem_param # Get the parameter from setup

        alpha = jax.nn.sigmoid(tau_mem) # This is the unique part of LICellSigmoid

        return li_update(x=in_sum, u=u, alpha=alpha)


class LICellBP(LICell):
    bit_precision: int = 32

    # This class also inherits its `setup` method from LICell.
    # Thus, `self.linear` and `self.tau_mem_param` are already defined.

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        # Reusing `self.linear` from parent `LICell`'s setup
        in_sum = self.linear(x)

        # Access the parameter created in parent `LICell`'s setup
        tau_mem = jnp.abs(self.tau_mem_param)

        alpha = jnp.exp(-1.0 / jnp.abs(tau_mem))
        alpha = quantize_tensor(alpha, self.bit_precision)

        u = li_update(x=in_sum, u=u, alpha=alpha)
        return u