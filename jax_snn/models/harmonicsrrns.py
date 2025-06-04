import jax
import jax.numpy as jnp
from flax import linen as nn
from .. import modules

class SimpleHarmonicRNN(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int
    adaptive_omega_a: float
    adaptive_omega_b: float
    adaptive_b_offset_a: float
    adaptive_b_offset_b: float
    out_adaptive_tau_mem_mean: float
    out_adaptive_tau_mem_std: float
    label_last: bool
    hidden_bias: bool = False
    output_bias: bool = False

    def setup(self):
        self.hidden = modules.HRFCell(
            input_size=self.input_size + self.hidden_size,  # recurrency
            layer_size=self.hidden_size,
            adaptive_omega=True,
            adaptive_omega_a=self.adaptive_omega_a,
            adaptive_omega_b=self.adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=self.adaptive_b_offset_a,
            adaptive_b_offset_b=self.adaptive_b_offset_b,
            bias=self.hidden_bias
        )

        self.out = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

        self.scanned_core = nn.scan(
            self.HarmonicScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden, self.out)

    def __call__(self, x):
        batch_size = x.shape[1]

        init_hidden_z = jnp.zeros((batch_size, self.hidden_size))
        init_hidden_u = jnp.zeros_like(init_hidden_z)
        init_hidden_v = jnp.zeros_like(init_hidden_z)
        init_hidden_a = jnp.zeros_like(init_hidden_z)
        init_out_u = jnp.zeros((batch_size, self.output_size))
        init_num_spikes = jnp.array(0., dtype=jnp.float32)

        init_carry = (
            init_hidden_z,
            init_hidden_u,
            init_hidden_v,
            init_hidden_a,
            init_out_u,
            init_num_spikes
        )

        (final_hidden_z, final_hidden_u, final_hidden_v, final_hidden_a, final_out_u,
         total_spikes), outputs_u = self.scanned_core(init_carry, x)

        if self.label_last:
            outputs = outputs_u[-1:, :, :]  # Retained original logic: -1:, :, :
        else:
            outputs = outputs_u

        return outputs, ((final_hidden_z, final_hidden_u, final_hidden_v, final_hidden_a), final_out_u), total_spikes

    class HarmonicScanCore(nn.Module):
        hidden: nn.Module
        out: nn.Module

        @nn.compact
        def __call__(self, carry, input_t):
            hidden_z, hidden_u, hidden_v, hidden_a, out_u, num_spikes = carry

            new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=-1),
                (hidden_z, hidden_u, hidden_v, hidden_a)
            )

            new_spikes = jnp.sum(new_hidden_z)
            new_out_u = self.out(new_hidden_z, out_u)

            new_num_spikes = num_spikes + new_spikes

            new_carry = (new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_a, new_out_u, new_num_spikes)
            return new_carry, new_out_u