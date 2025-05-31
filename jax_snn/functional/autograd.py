

import jax
import jax.numpy as jnp
from typing import Tuple


from .base import *

p = 0.15
scale = 6.0
length = 0.5
gamma = 0.5
sigma1 = length
sigma2 = scale * length

@jax.custom_vjp
def StepGaussianGrad(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def StepGaussianGrad_fwd(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    y = step(x)
    return y, x

def StepGaussianGrad_bwd(x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray]:
    dfdx = gaussian(x)
    return (g * dfdx,)


# --- 2. StepLinearGrad ---
@jax.custom_vjp
def StepLinearGrad(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def StepLinearGrad_fwd(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    y = step(x)
    return y, x

def StepLinearGrad_bwd(x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray]:
    dfdx = jax.nn.relu(1.0 - jnp.abs(x))
    return (g * dfdx,)


# --- 3. StepExpGrad ---

@jax.custom_vjp
def StepExpGrad(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def StepExpGrad_fwd(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    y = step(x)
    return y, x

def StepExpGrad_bwd(x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray]:
    dfdx = jnp.exp(-jnp.abs(x))
    return (g * dfdx,)

# --- 4. StepDoubleGaussianGrad ---

@jax.custom_vjp
def StepDoubleGaussianGrad(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def StepDoubleGaussianGrad_fwd(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    y = step(x)
    return y, x

def StepDoubleGaussianGrad_bwd(x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray]:
    p = 0.15
    scale = 6.0
    length = 0.5
    gamma = 0.5

    sigma1 = length
    sigma2 = scale * length

    dfd = (1.0 + p) * gaussian(x, mu=0.0, sigma=sigma1) - 2.0 * p * gaussian(x, mu=0.0, sigma=sigma2)
    return (g * dfd * gamma,)



# --- 5. StepMultiGaussianGrad ---

@jax.custom_vjp
def StepMultiGaussianGrad(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def StepMultiGaussianGrad_fwd(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    y = step(x)
    return y, x

def StepMultiGaussianGrad_bwd(x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray]:
    dfd = (1.0 + p) * gaussian(x, mu=0.0, sigma=sigma1) \
          - p * gaussian(x, mu=length, sigma=sigma2) - p * gaussian(x, mu=-length, sigma=sigma2)
    return (g * dfd * gamma,)


@jax.jit
def FGI_DGaussian(x: jnp.ndarray) -> jnp.ndarray:
    x_detached = jax.lax.stop_gradient(step(x))

    df = (1.0 + p) * gaussian(x, mu=0.0, sigma=sigma1) - 2.0 * p * gaussian(x, mu=0.0, sigma=sigma2)

    df_detached = jax.lax.stop_gradient(df)

    dfd = gamma * df_detached * x

    dfd_detached = jax.lax.stop_gradient(dfd)

    return dfd - dfd_detached + x_detached


# DEFVJP
StepGaussianGrad.defvjp(StepGaussianGrad_fwd, StepGaussianGrad_bwd)
StepLinearGrad.defvjp(StepLinearGrad_fwd, StepLinearGrad_bwd)
StepExpGrad.defvjp(StepExpGrad_fwd, StepExpGrad_bwd)
StepDoubleGaussianGrad.defvjp(StepDoubleGaussianGrad_fwd, StepDoubleGaussianGrad_bwd)
StepMultiGaussianGrad.defvjp(StepMultiGaussianGrad_fwd, StepMultiGaussianGrad_bwd)
