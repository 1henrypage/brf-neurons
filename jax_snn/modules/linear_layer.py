import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import zeros, xavier_uniform


class LinearMask(nn.Module):
    in_features: int
    out_features: int
    bias: bool
    lbd: int
    ubd: int
    mask_prob: float = 0.0

    def setup(self):
        self.weight = self.param(
            'weight',
            xavier_uniform(),
            (self.out_features, self.in_features)
        )

        if self.bias:
            self.bias_param = self.param(
                'bias',
                zeros,
                (self.out_features,)
            )
        else:
            self.bias_param = None

        key = self.make_rng('params')
        mask = jnp.ones((self.out_features, self.in_features))

        if self.mask_prob > 0:
            rand_vals = jax.random.uniform(key, (self.out_features, self.ubd - self.lbd))
            masked_region = rand_vals > self.mask_prob
            mask = mask.at[:, self.lbd:self.ubd].multiply(masked_region.astype(mask.dtype))

        self.mask = self.variable('constants', 'mask', lambda: mask)

    def __call__(self, x):
        masked_weight = self.weight * self.mask.value
        y = jnp.dot(x, masked_weight.T)
        if self.bias_param is not None:
            y += self.bias_param
        return y
