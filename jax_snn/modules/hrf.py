import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

DEFAULT_MASK_PROB = 0

TRAIN_B_offset = True
DEFAULT_RF_B_offset = 1.

DEFAULT_RF_ADAPTIVE_B_offset_a = 1
DEFAULT_RF_ADAPTIVE_B_offset_b = 6

TRAIN_OMEGA = True
DEFAULT_RF_OMEGA = 10.

DEFAULT_RF_ADAPTIVE_OMEGA_a = 10
DEFAULT_RF_ADAPTIVE_OMEGA_b = 50

DEFAULT_RF_THETA = 1

TRAIN_ZETA = False
DEFAULT_RF_ZETA = .00

DEFAULT_RF_ADAPTIVE_ZETA_a = 0
DEFAULT_RF_ADAPTIVE_ZETA_b = 0

TRAIN_DT = False
DEFAULT_DT = 0.01
DEFAULT_RF_ADAPTIVE_DT = 0.01

# ---------------------------------------------------------------
# Functional part
# ---------------------------------------------------------------

@jax.jit
def hrf_update(
        x: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        ref_period: jnp.ndarray,
        b: jnp.ndarray,
        omega: jnp.ndarray,
        dt: float = DEFAULT_DT,
        theta: float = DEFAULT_RF_THETA,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    v = v + u * dt
    u = u + x * dt - b * u * (2 * dt) - (omega ** 2) * v * dt

    z = jnp.where((u - theta - ref_period) > 0, 1.0, 0.0)
    ref_period = ref_period * 0.9 + z

    return z, u, v, ref_period

# ---------------------------------------------------------------
# HRFCell (JAX/Flax Module)
# ---------------------------------------------------------------

class HRFCell(nn.Module):
    input_size: int
    layer_size: int
    mask_prob: float = DEFAULT_MASK_PROB
    b_offset: float = DEFAULT_RF_B_offset
    adaptive_b_offset: bool = TRAIN_B_offset
    adaptive_b_offset_a: float = DEFAULT_RF_ADAPTIVE_B_offset_a
    adaptive_b_offset_b: float = DEFAULT_RF_ADAPTIVE_B_offset_b
    omega: float = DEFAULT_RF_OMEGA
    adaptive_omega: bool = TRAIN_OMEGA
    adaptive_omega_a: float = DEFAULT_RF_ADAPTIVE_OMEGA_a
    adaptive_omega_b: float = DEFAULT_RF_ADAPTIVE_OMEGA_b
    dt: float = DEFAULT_DT
    bias: bool = False

    def setup(self):
        # Linear layer
        self.linear = nn.Dense(
            features=self.layer_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform()
        )

        # Omega initialization
        omega_init = jnp.ones((self.layer_size,)) * self.omega
        if self.adaptive_omega:
            self.omega_param = self.param(
                'omega',
                lambda key, shape: jax.random.uniform(
                    key, shape, minval=self.adaptive_omega_a, maxval=self.adaptive_omega_b
                ),
                omega_init.shape
            )
        else:
            self.omega_param = omega_init

        # b_offset initialization
        b_offset_init = jnp.ones((self.layer_size,)) * self.b_offset
        if self.adaptive_b_offset:
            self.b_offset_param = self.param(
                'b_offset',
                lambda key, shape: jax.random.uniform(
                    key, shape, minval=self.adaptive_b_offset_a, maxval=self.adaptive_b_offset_b
                ),
                b_offset_init.shape
            )
        else:
            self.b_offset_param = b_offset_init

    def __call__(
            self,
            x: jnp.ndarray,
            state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        z, u, v, ref_period = state

        in_sum = self.linear(x)

        omega = jnp.abs(self.omega_param)
        b_offset = jnp.abs(self.b_offset_param)

        b = (omega ** 2) * 0.005 + b_offset + ref_period

        z, u, v, ref_period = hrf_update(
            x=in_sum,
            u=u,
            v=v,
            ref_period=ref_period,
            b=b,
            omega=omega,
            dt=self.dt
        )

        return z, u, v, ref_period
