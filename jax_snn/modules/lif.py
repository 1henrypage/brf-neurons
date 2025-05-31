import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import normal, zeros, lecun_normal, variance_scaling
from typing import Tuple

from torch.nn.init import xavier_uniform

from jax_snn.functional import StepLinearGrad, quantize_tensor
from jax_snn.modules import LinearMask

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

    @nn.compact
    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        linear = nn.Dense(
            features=self.layer_size,
            use_bias=self.bias,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
        )
        in_sum = linear(x)

        if self.adaptive_tau_mem:
            tau_mem_param = self.param(
                "tau_mem",
                lambda key, shape: normal(stddev=self.adaptive_tau_mem_std)(key, shape) + self.adaptive_tau_mem_mean,
                (self.layer_size,),
            )
            tau_mem = jnp.abs(tau_mem_param)
        else:
            tau_mem = jnp.full((self.layer_size,), self.tau_mem)

        tau_mem = jnp.abs(tau_mem)


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

    @nn.compact
    def __call__(self, x: jnp.ndarray, state: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z, u = state

        linear = nn.Dense(
            features=self.layer_size,
            use_bias=self.bias,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
        )
        in_sum = linear(x)

        if self.adaptive_tau_mem:
            tau_mem_param = self.param(
                "tau_mem",
                lambda key, shape: normal(stddev=self.adaptive_tau_mem_std)(key, shape) + self.adaptive_tau_mem_mean,
                (self.layer_size,),
            )
            tau_mem = jnp.abs(tau_mem_param)
        else:
            tau_mem = jnp.full((self.layer_size,), self.tau_mem)

        alpha = jnp.sigmoid(-1.0 / jnp.abs(tau_mem))

        z, u = lif_update(x=in_sum, u=u, alpha=alpha)
        return z, u


class LICellSigmoid(LICell):
    @nn.compact
    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        w_init = nn.initializers.xavier_uniform()
        b_init = nn.initializers.zeros if self.use_bias else None

        dense = nn.Dense(self.layer_size, use_bias=self.use_bias, kernel_init=w_init, bias_init=b_init)
        in_sum = dense(x)

        tau_mem = self.param('tau_mem', lambda key, shape:
        jax.random.normal(key, shape) * self.adaptive_tau_mem_std + self.adaptive_tau_mem_mean,
                             (self.layer_size,)) if self.adaptive_tau_mem else self.tau_mem * jnp.ones(self.layer_size)

        alpha = jax.nn.sigmoid(tau_mem)

        return li_update(x=in_sum, u=u, alpha=alpha)




class LICellBP(LICell):
    bit_precision: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        linear = nn.Dense(
            features=self.layer_size,
            use_bias=self.bias,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
        )
        in_sum = linear(x)

        if self.adaptive_tau_mem:
            tau_mem_param = self.param(
                "tau_mem",
                lambda key, shape: normal(stddev=self.adaptive_tau_mem_std)(key, shape) + self.adaptive_tau_mem_mean,
                (self.layer_size,),
            )
            tau_mem = jnp.abs(tau_mem_param)
        else:
            tau_mem = jnp.full((self.layer_size,), self.tau_mem)

        alpha = jnp.exp(-1.0 / jnp.abs(tau_mem))
        alpha = quantize_tensor(alpha, self.bit_precision)

        u = li_update(x=in_sum, u=u, alpha=alpha)
        return u
