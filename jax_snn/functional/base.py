import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# Precompute constant sqrt(2pi) once for efficiency
_INV_SQRT_2PI = 1.0 / jnp.sqrt(2 * jnp.pi)

@jax.jit
def step(x: jnp.ndarray) -> jnp.ndarray:
    return (x > 0).astype(jnp.float32)


@jax.jit
def exp_decay(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-jnp.abs(x))


@partial(jax.jit, static_argnums=(1, 2))
def gaussian(x: jnp.ndarray, mu: float = 0.0, sigma: float = 1.0) -> jnp.ndarray:
    scale = _INV_SQRT_2PI / sigma
    arg = (x - mu) / sigma
    return scale * jnp.exp(-0.5 * arg * arg)

@partial(jax.jit, static_argnums=(1, 2))
def gaussian_non_normalized(x: jnp.ndarray, mu: float = 0.0, sigma: float = 1.0) -> jnp.ndarray:
    arg = (x - mu) / sigma
    return jnp.exp(-0.5 * arg * arg)

@jax.jit
def std_gaussian(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-0.5 * x ** 2)

@jax.jit
def linear_peak(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.relu(1.0 - jnp.abs(x))

@jax.jit
def linear_peak_antiderivative(x: jnp.ndarray) -> jnp.ndarray:
    xa = jax.nn.relu(1.0 - jnp.abs(x))
    xa_sq = xa ** 2
    # Use lax.cond to avoid Python branching inside jitted function
    return 0.5 * lax.select(x < 0, xa_sq, 2.0 - xa_sq)


@jax.jit
def DoubleGaussian(x: jnp.ndarray) -> jnp.ndarray:
    p = 0.15
    scale = 6.0
    length = 0.5
    gamma = 0.5
    sigma1 = length
    sigma2 = scale * length
    return gamma * (1. + p) * gaussian(x, mu=0., sigma=sigma1) \
        - p * gaussian(x, mu=length, sigma=sigma2) \
        - p * gaussian(x, mu=-length, sigma=sigma2)


@partial(jax.jit, static_argnums=(1,))
def quantize_tensor(tensor: jnp.ndarray, f: int) -> jnp.ndarray:
    factor = 2 ** f
    return jnp.round(factor * tensor) * (1.0 / factor)


@partial(jax.jit, static_argnums=(2,))
def spike_deletion(hidden_z: jnp.ndarray, spike_del_p: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    # Generate uniform random values in [0,1) with the same shape as hidden_z
    rand_vals = jax.random.uniform(key, shape=hidden_z.shape)
    mask = spike_del_p < rand_vals
    return hidden_z * mask.astype(hidden_z.dtype)



