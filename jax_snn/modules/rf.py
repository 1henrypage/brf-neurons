import jax.numpy as jnp
import jax
from flax import linen as nn

from jax import random
from typing import Tuple, Optional

from jax_snn.functional import FGI_DGaussian, StepDoubleGaussianGrad
from jax_snn.modules.linear_layer import LinearMask

DEFAULT_MASK_PROB = 0.0
TRAIN_B_offset = True
DEFAULT_RF_B_offset = 1.0
DEFAULT_RF_ADAPTIVE_B_offset_a = 0.0
DEFAULT_RF_ADAPTIVE_B_offset_b = 3.0
TRAIN_OMEGA = True
DEFAULT_RF_OMEGA = 10.0
DEFAULT_RF_ADAPTIVE_OMEGA_a = 5.0
DEFAULT_RF_ADAPTIVE_OMEGA_b = 10.0
DEFAULT_RF_THETA = 1.0
DEFAULT_DT = 0.01


def rf_update(
        x: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        b: jnp.ndarray,
        omega: jnp.ndarray,
        dt: float = DEFAULT_DT,
        theta: float = DEFAULT_RF_THETA,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u_ = u + b * u * dt - omega * v * dt + x * dt
    v = v + omega * u * dt + b * v * dt
    z = FGI_DGaussian(u_ - theta)
    return z, u_, v


def brf_update(
        x: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
        omega: jnp.ndarray,
        dt: float = DEFAULT_DT,
        theta: float = DEFAULT_RF_THETA,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u_ = u + b * u * dt - omega * v * dt + x * dt
    v = v + omega * u * dt + b * v * dt
    z = StepDoubleGaussianGrad(u_ - theta - q)
    q = q * 0.9 + z
    return z, u_, v, q


def izhikevich_update(
        x: jnp.ndarray,
        u: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
        omega: jnp.ndarray,
        dt: float = DEFAULT_DT,
        theta: float = DEFAULT_RF_THETA,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u = u + u * (b + 1j * omega) * dt + x * dt
    z = StepDoubleGaussianGrad(u.imag - theta)
    u = u - u * z + z * 1j
    return z, u, q


def sustain_osc(omega: jnp.ndarray, dt: float = DEFAULT_DT) -> jnp.ndarray:
    return (-1 + jnp.sqrt(1 - (dt * omega) ** 2)) / dt


class RFCell(nn.Module):
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
    pruning: bool = False

    def setup(self):
        if self.pruning:
            self.linear = LinearMask(
                in_features=self.input_size,
                out_features=self.layer_size,
                bias=self.bias,
                mask_prob=self.mask_prob,
                lbd=self.input_size - self.layer_size,
                ubd=self.input_size,
            )
        else:
            self.linear = nn.Dense(
                features=self.layer_size,
                use_bias=self.bias,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )

        if self.adaptive_omega:
            self.omega_param = self.param(
                "omega",
                lambda rng, shape: jax.random.uniform(rng, shape, minval=self.adaptive_omega_a, maxval=self.adaptive_omega_b),
                (self.layer_size,),
            )
        else:
            self.omega_param = jnp.full((self.layer_size,), self.omega)

        if self.adaptive_b_offset:
            self.b_offset_param = self.param(
                "b_offset",
                lambda rng, shape: jax.random.uniform(rng, shape, minval=self.adaptive_b_offset_a,
                                                      maxval=self.adaptive_b_offset_b),
                (self.layer_size,),
            )
        else:
            self.b_offset_param = jnp.full((self.layer_size,), self.b_offset)

    def __call__(
            self,
            x: jnp.ndarray,
            state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z, u, v = state

        in_sum = self.linear(x)

        omega = jnp.abs(self.omega_param)
        b = -jnp.abs(self.b_offset_param)


        z, u, v = rf_update(
            x=in_sum,
            u=u,
            v=v,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        # clamp idk wtf was going wrong here, but u and v were exploding on SMNIST
        u = jnp.clip(u, -1e6, 1e6)
        v = jnp.clip(v, -1e6, 1e6)

        # Debug: post-update

        return z, u, v




class BRFCell(RFCell):

    def __call__(
            self,
            x: jnp.ndarray,
            state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        in_sum = self.linear(x)
        z, u, v, q = state

        omega = jnp.abs(self.omega_param)
        b_offset = jnp.abs(self.b_offset_param)

        p_omega = sustain_osc(omega)
        b = p_omega - b_offset - q

        z, u, v, q = brf_update(
            x=in_sum,
            u=u,
            v=v,
            q=q,
            b=b,
            omega=omega,
            dt=self.dt,
        )
        return z, u, v, q

