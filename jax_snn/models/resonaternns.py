import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
from .. import modules
from ..functional import spike_deletion, quantize_tensor


class SimpleResRNN(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int
    pruning: bool = False
    adaptive_omega_a: float = 5.0
    adaptive_omega_b: float = 10.0
    adaptive_b_offset_a: float = 0.0
    adaptive_b_offset_b: float = 1.0
    out_adaptive_tau_mem_mean: float = 20.0
    out_adaptive_tau_mem_std: float = 5.0
    n_last: int = 1
    mask_prob: float = 0.0
    sub_seq_length: int = 0
    hidden_bias: bool = False
    output_bias: bool = False
    label_last: bool = False
    dt: float = 0.01

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        hidden = modules.BRFCell(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            bias=self.hidden_bias,
            mask_prob=self.mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=self.adaptive_omega_a,
            adaptive_omega_b=self.adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=self.adaptive_b_offset_a,
            adaptive_b_offset_b=self.adaptive_b_offset_b,
            dt=self.dt,
            pruning=self.pruning
        )

        out = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

        def loop_fn(carry, x_t):
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            hidden_z, hidden_u, hidden_v, hidden_q = hidden(x_in, (hidden_z, hidden_u, hidden_v, hidden_q), train=train)
            spike_sum += jnp.sum(hidden_z)
            out_u = out(hidden_z, out_u, train=train)
            return (hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum), out_u

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_h, init_out, jnp.array(0.0))

        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = jax.lax.scan(loop_fn, init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-self.n_last:]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum


class SimpleVanillaRFRNN(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int
    pruning: bool = False
    adaptive_omega_a: float = 5.0
    adaptive_omega_b: float = 10.0
    adaptive_b_offset_a: float = 0.0
    adaptive_b_offset_b: float = 1.0
    out_adaptive_tau_mem_mean: float = 20.0
    out_adaptive_tau_mem_std: float = 5.0
    n_last: int = 1
    mask_prob: float = 0.0
    sub_seq_length: int = 0
    hidden_bias: bool = False
    output_bias: bool = False
    label_last: bool = False
    dt: float = 0.01

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        hidden = modules.RFCell(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            bias=self.hidden_bias,
            mask_prob=self.mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=self.adaptive_omega_a,
            adaptive_omega_b=self.adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=self.adaptive_b_offset_a,
            adaptive_b_offset_b=self.adaptive_b_offset_b,
            dt=self.dt,
            pruning=self.pruning
        )

        out = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

        def loop_fn(carry, x_t):
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            hidden_z, hidden_u, hidden_v = hidden(x_in, (hidden_z, hidden_u, hidden_v), train=train)
            spike_sum += jnp.sum(hidden_z)
            out_u = out(hidden_z, out_u, train=train)
            return (hidden_z, hidden_u, hidden_v, out_u, spike_sum), out_u

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_out, jnp.array(0.0))

        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = jax.lax.scan(loop_fn, init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-self.n_last:]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum



class BRFRSNN_SD(SimpleResRNN):
    spike_del_p: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        hidden = modules.BRFCell(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            bias=self.hidden_bias,
            mask_prob=self.mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=self.adaptive_omega_a,
            adaptive_omega_b=self.adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=self.adaptive_b_offset_a,
            adaptive_b_offset_b=self.adaptive_b_offset_b,
            dt=self.dt,
            pruning=self.pruning
        )

        out = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

        def loop_fn(carry, x_t):
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            hidden_z, hidden_u, hidden_v, hidden_q = hidden(x_in, (hidden_z, hidden_u, hidden_v, hidden_q), train=train)

            # spike deletion
            hidden_z = spike_deletion(hidden_z=hidden_z,
                                      spike_del_p=self.spike_del_p,
                                      key=self.make_rng('spike_deletion')
                                      )

            spike_sum += jnp.sum(hidden_z)
            out_u = out(hidden_z, out_u, train=train)
            return (hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum), out_u

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_h, init_out, jnp.array(0.0))

        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = jax.lax.scan(loop_fn, init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum


class BRFRSNN_BP(SimpleResRNN):
    bit_precision: int = 52

    def setup(self):
        super().setup()
        self.out = modules.LICellBP(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            bit_precision=self.bit_precision,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        x = quantize_tensor(x, self.bit_precision)

        hidden_z = quantize_tensor(jnp.zeros((batch_size, self.hidden_size)), self.bit_precision)
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_v = jnp.zeros_like(hidden_z)
        hidden_q = jnp.zeros_like(hidden_z)

        out_u = quantize_tensor(jnp.zeros((batch_size, self.output_size)), self.bit_precision)

        hidden = modules.BRFCell(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            bias=self.hidden_bias,
            mask_prob=self.mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=self.adaptive_omega_a,
            adaptive_omega_b=self.adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=self.adaptive_b_offset_a,
            adaptive_b_offset_b=self.adaptive_b_offset_b,
            dt=self.dt,
            pruning=self.pruning
        )

        def loop_fn(carry, x_t):
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            hidden_z, hidden_u, hidden_v, hidden_q = hidden(x_in, (hidden_z, hidden_u, hidden_v, hidden_q), train=train)

            spike_sum += jnp.sum(hidden_z)
            out_u = self.out(hidden_z, out_u, train=train)
            return (hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum), out_u

        init_state = (hidden_z, hidden_u, hidden_v, hidden_q, out_u, jnp.array(0.0))

        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = jax.lax.scan(loop_fn, init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum


class RFRSNN_SD(SimpleVanillaRFRNN):
    spike_del_p: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        hidden = modules.RFCell(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            bias=self.hidden_bias,
            mask_prob=self.mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=self.adaptive_omega_a,
            adaptive_omega_b=self.adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=self.adaptive_b_offset_a,
            adaptive_b_offset_b=self.adaptive_b_offset_b,
            dt=self.dt,
            pruning=self.pruning
        )

        out = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

        def loop_fn(carry, x_t):
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            hidden_z, hidden_u, hidden_v = hidden(x_in, (hidden_z, hidden_u, hidden_v), train=train)

            hidden_z = spike_deletion(hidden_z=hidden_z,
                                      spike_del_p=self.spike_del_p,
                                      key=self.make_rng('spike_deletion')
                                      )

            spike_sum += jnp.sum(hidden_z)
            out_u = out(hidden_z, out_u, train=train)
            return (hidden_z, hidden_u, hidden_v, out_u, spike_sum), out_u

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_out, jnp.array(0.0))

        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = jax.lax.scan(loop_fn, init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum


class RFRSNN_BP(SimpleVanillaRFRNN):
    bit_precision: int = 32

    def setup(self):
        super().setup()
        self.out = modules.LICellBP(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            bit_precision=self.bit_precision,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        x = quantize_tensor(x, self.bit_precision)
        hidden_z = quantize_tensor(jnp.zeros((batch_size, self.hidden_size)), self.bit_precision)
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_v = jnp.zeros_like(hidden_z)
        out_u = quantize_tensor(jnp.zeros((batch_size, self.output_size)), self.bit_precision)

        hidden = modules.RFCell(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            bias=self.hidden_bias,
            mask_prob=self.mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=self.adaptive_omega_a,
            adaptive_omega_b=self.adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=self.adaptive_b_offset_a,
            adaptive_b_offset_b=self.adaptive_b_offset_b,
            dt=self.dt,
            pruning=self.pruning
        )

        def loop_fn(carry, x_t):
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            hidden_z, hidden_u, hidden_v = hidden(x_in, (hidden_z, hidden_u, hidden_v), train=train)
            spike_sum += jnp.sum(hidden_z)
            out_u = self.out(hidden_z, out_u, train=train)
            return (hidden_z, hidden_u, hidden_v, out_u, spike_sum), out_u

        init_state = (hidden_z, hidden_u, hidden_v, out_u, jnp.array(0.0))

        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = jax.lax.scan(loop_fn, init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum
