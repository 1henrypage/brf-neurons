from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from .. import modules
from ..functional import spike_deletion, quantize_tensor
import optax

# alpha parameter initialization
DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_ALIF_ADAPTIVE_TAU_M_STD = 5.


# rho parameter initialization
DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN = 20.
DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD = 5.

class SimpleALIFRNN(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int
    pruning: bool = False
    adaptive_tau_mem: bool = True  # adaptive time constant for alpha
    adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN
    adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD
    adaptive_tau_adp: bool = True  # adaptive time constant for rho
    adaptive_tau_adp_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN
    adaptive_tau_adp_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD
    out_adaptive_tau: bool = True  # adaptive time constant for LI output
    out_adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN
    out_adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD
    hidden_bias: bool = False
    output_bias: bool = False
    sub_seq_length: int = 0
    mask_prob: float = 0.
    label_last: bool = False
    n_last: int = 1

    def setup(self):
        self.hidden = modules.ALIFCell(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            adaptive_tau_mem=self.adaptive_tau_mem,
            adaptive_tau_mem_mean=self.adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.adaptive_tau_mem_std,
            adaptive_tau_adp=self.adaptive_tau_adp,
            adaptive_tau_adp_mean=self.adaptive_tau_adp_mean,
            adaptive_tau_adp_std=self.adaptive_tau_adp_std,
            bias=self.hidden_bias,
            mask_prob=self.mask_prob,
            pruning=self.pruning
        )

        self.out = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=self.out_adaptive_tau,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

    def __call__(self, x):
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        hidden_z = jnp.zeros((batch_size, self.hidden_size))
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_a = jnp.zeros_like(hidden_z)
        out_u = jnp.zeros((batch_size, self.output_size))
        num_spikes = jnp.array(0.0, dtype=jnp.float32)

        def scan_fn(carry, input_t):
            hidden_z, hidden_u, hidden_a, out_u, num_spikes = carry
            hidden = (hidden_z, hidden_u, hidden_a)

            hidden_z, hidden_u, hidden_a = self.hidden(
                # this can be problematic
                jnp.concatenate((input_t, hidden_z), axis=1),
                hidden
            )

            out_u = self.out(hidden_z, out_u)
            num_spikes = num_spikes + jnp.sum(hidden_z)

            return (hidden_z, hidden_u, hidden_a, out_u, num_spikes), out_u

        init_carry = (hidden_z, hidden_u, hidden_a, out_u, num_spikes)

        (final_hidden_z, final_hidden_u, final_hidden_a, final_out_u, total_spikes), outputs = jax.lax.scan(
            scan_fn,
            init_carry,
            x
        )

        if self.sub_seq_length > 0:
            outputs = outputs[self.sub_seq_length:]

        if self.label_last:
            outputs = outputs[-self.n_last:]

        return outputs, ((final_hidden_z, final_hidden_u, final_hidden_a), final_out_u), total_spikes

class DoubleALIFRNN(nn.Module):
    input_size: int
    hidden1_size: int
    hidden2_size: int
    output_size: int
    adaptive_tau_mem: bool = True
    adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN
    adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD
    adaptive_tau_adp: bool = True
    adaptive_tau_adp_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN
    adaptive_tau_adp_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD
    out_adaptive_tau: bool = True
    out_adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN
    out_adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD
    hidden1_bias: bool = False
    hidden2_bias: bool = False
    output_bias: bool = False
    sub_seq_length: int = 0
    mask_prob: float = 0.
    label_last: bool = False

    def setup(self):
        self.hidden1 = modules.ALIFCell(
            input_size=self.input_size + self.hidden1_size,
            layer_size=self.hidden1_size,
            adaptive_tau_mem=self.adaptive_tau_mem,
            adaptive_tau_mem_mean=self.adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.adaptive_tau_mem_std,
            adaptive_tau_adp=self.adaptive_tau_adp,
            adaptive_tau_adp_mean=self.adaptive_tau_adp_mean,
            adaptive_tau_adp_std=self.adaptive_tau_adp_std,
            bias=self.hidden1_bias,
            mask_prob=self.mask_prob
        )
        self.hidden2 = modules.ALIFCell(
            input_size=self.hidden1_size + self.hidden2_size,
            layer_size=self.hidden2_size,
            adaptive_tau_mem=self.adaptive_tau_mem,
            adaptive_tau_mem_mean=self.adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.adaptive_tau_mem_std,
            adaptive_tau_adp=self.adaptive_tau_adp,
            adaptive_tau_adp_mean=self.adaptive_tau_adp_mean,
            adaptive_tau_adp_std=self.adaptive_tau_adp_std,
            bias=self.hidden2_bias,
            mask_prob=self.mask_prob
        )
        self.out = modules.LICell(
            input_size=self.hidden2_size,
            layer_size=self.output_size,
            adaptive_tau_mem=self.out_adaptive_tau,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

    def __call__(self, x):
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        hidden1_z = jnp.zeros((batch_size, self.hidden1_size))
        hidden1_u = jnp.zeros_like(hidden1_z)
        hidden1_a = jnp.zeros_like(hidden1_z)

        hidden2_z = jnp.zeros((batch_size, self.hidden2_size))
        hidden2_u = jnp.zeros_like(hidden2_z)
        hidden2_a = jnp.zeros_like(hidden2_z)

        out_u = jnp.zeros((batch_size, self.output_size))

        def scan_fn(carry, input_t):
            hidden1_z, hidden1_u, hidden1_a, hidden2_z, hidden2_u, hidden2_a, out_u = carry

            hidden1 = (hidden1_z, hidden1_u, hidden1_a)
            hidden1_z, hidden1_u, hidden1_a = self.hidden1(
                jnp.concatenate((input_t, hidden1_z), axis=1),
                hidden1
            )

            hidden2 = (hidden2_z, hidden2_u, hidden2_a)
            hidden2_z, hidden2_u, hidden2_a = self.hidden2(
                jnp.concatenate((hidden1_z, hidden2_z), axis=1),
                hidden2
            )

            out_u = self.out(hidden2_z, out_u)

            carry = (hidden1_z, hidden1_u, hidden1_a, hidden2_z, hidden2_u, hidden2_a, out_u)
            return carry, out_u

        init_carry = (
            hidden1_z, hidden1_u, hidden1_a,
            hidden2_z, hidden2_u, hidden2_a,
            out_u
        )

        carry, outputs = jax.lax.scan(scan_fn, init_carry, x)

        hidden1_z, hidden1_u, hidden1_a, hidden2_z, hidden2_u, hidden2_a, out_u = carry

        if self.sub_seq_length > 0:
            outputs = outputs[self.sub_seq_length + 1:]

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((hidden1_z, hidden1_u, hidden1_a, hidden2_z, hidden2_u, hidden2_a), out_u)

class ALIFRSNN_SD(SimpleALIFRNN):
    spike_del_p: float = 0.
    def __call__(self, x):
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        hidden_z = jnp.zeros((batch_size, self.hidden_size))
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_a = jnp.zeros_like(hidden_z)
        out_u = jnp.zeros((batch_size, self.output_size))
        num_spikes = jnp.array(0., dtype=jnp.float32)

        def scan_fn(carry, input_t):
            hidden_z, hidden_u, hidden_a, out_u, num_spikes = carry
            hidden = (hidden_z, hidden_u, hidden_a)

            hidden_z, hidden_u, hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=1),
                hidden
            )

            hidden_z = spike_deletion(
                hidden_z=hidden_z,
                spike_del_p=self.spike_del_p,
                key=self.make_rng('spike_deletion'),
            )

            num_spikes = num_spikes + jnp.sum(hidden_z)
            out_u = self.out(hidden_z, out_u)

            return (hidden_z, hidden_u, hidden_a, out_u, num_spikes), out_u

        init_carry = (hidden_z, hidden_u, hidden_a, out_u, num_spikes)

        (final_hidden_z, final_hidden_u, final_hidden_a, final_out_u, total_spikes), outputs = jax.lax.scan(
            scan_fn,
            init_carry,
            x
        )

        if self.sub_seq_length > 0:
            outputs = outputs[self.sub_seq_length:]

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((final_hidden_z, final_hidden_u, final_hidden_a), final_out_u), total_spikes


class ALIFRSNN_BP(SimpleALIFRNN):
    def __call__(self, x):
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        x = quantize_tensor(x, self.bit_precision)
        hidden_z = quantize_tensor(jnp.zeros((batch_size, self.hidden_size)), self.bit_precision)
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_a = jnp.zeros_like(hidden_z)
        out_u = quantize_tensor(jnp.zeros((batch_size, self.output_size)), self.bit_precision)

        num_spikes = jnp.array(0., dtype=jnp.float32)

        def scan_fn(carry, input_t):
            hidden_z, hidden_u, hidden_a, out_u, num_spikes = carry
            hidden = (hidden_z, hidden_u, hidden_a)

            hidden_z, hidden_u, hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=1),
                hidden
            )

            num_spikes = num_spikes + jnp.sum(hidden_z)
            out_u = self.out(hidden_z, out_u)

            return (hidden_z, hidden_u, hidden_a, out_u, num_spikes), out_u

        init_carry = (hidden_z, hidden_u, hidden_a, out_u, num_spikes)

        (final_hidden_z, final_hidden_u, final_hidden_a, final_out_u, total_spikes), outputs = jax.lax.scan(
            scan_fn,
            init_carry,
            x
        )

        if self.sub_seq_length > 0:
            outputs = outputs[self.sub_seq_length:]

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((final_hidden_z, final_hidden_u, final_hidden_a), final_out_u), total_spikes

class SimpleALIFRNNTbptt(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int
    criterion: Callable

    pruning: bool = False
    adaptive_tau_mem: bool = True
    adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN
    adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD
    adaptive_tau_adp: bool = True
    adaptive_tau_adp_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN
    adaptive_tau_adp_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD
    out_adaptive_tau: bool = True
    out_adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN
    out_adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD
    hidden_bias: bool = False
    output_bias: bool = False

    tbptt_steps: int = 50
    sub_seq_length: int = 0
    mask_prob: float = 0.0
    label_last: bool = False
    n_last: int = 1

    def setup(self):
        self.hidden = modules.ALIFCell(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            adaptive_tau_mem=self.adaptive_tau_mem,
            adaptive_tau_mem_mean=self.adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.adaptive_tau_mem_std,
            adaptive_tau_adp=self.adaptive_tau_adp,
            adaptive_tau_adp_mean=self.adaptive_tau_adp_mean,
            adaptive_tau_adp_std=self.adaptive_tau_adp_std,
            bias=self.hidden_bias,
            mask_prob=self.mask_prob,
            pruning=self.pruning,
        )

        self.out = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=self.out_adaptive_tau,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias,
        )

    def __call__(self, x, y, params, opt_state, optimizer, train: bool = True):
        sequence_length, batch_size = x.shape[0], x.shape[1]

        def step_fn(carry, t):
            (
                hidden_z,
                hidden_u,
                hidden_a,
                out_u,
                loss_val,
                total_loss,
                total_loss_ll,
                num_spikes,
                params,
                opt_state,
            ) = carry

            input_t = x[t]
            target_t = y[t]

            hidden_input = jnp.concatenate([input_t, hidden_z], axis=1)
            hidden_z_new, (hidden_u_new, hidden_a_new) = self.hidden.apply(
                params["hidden"],
                hidden_input,
                (hidden_z, hidden_u, hidden_a),
                mutable=False,
                rngs=None
            )
            out_u_new = self.out.apply(params["out"], hidden_z_new, out_u)

            spikes = jnp.sum(hidden_z_new)
            num_spikes += spikes

            def compute_loss_and_update():
                if hasattr(self.criterion, '__name__') and self.criterion.__name__ == 'nll_loss':
                    logits = nn.log_softmax(out_u_new, axis=1)
                else:
                    logits = out_u_new

                loss = self.criterion(logits, target_t)
                new_loss_val = loss_val + loss
                new_total_loss = total_loss + loss

                def apply_grads():
                    def loss_fn(p):
                        h_z, (h_u, h_a) = self.hidden.apply(
                            p["hidden"], hidden_input, (hidden_z, hidden_u, hidden_a), mutable=False
                        )
                        o_u = self.out.apply(p["out"], h_z, out_u)
                        lgt = nn.log_softmax(o_u, axis=1) if hasattr(self.criterion, '__name__') and self.criterion.__name__ == 'nll_loss' else o_u
                        return self.criterion(lgt, target_t)

                    grads = jax.grad(loss_fn)(params)
                    updates, new_opt_state = optimizer.update(grads, opt_state)
                    new_params = optax.apply_updates(params, updates)

                    return (
                        jax.lax.stop_gradient(hidden_z_new),
                        jax.lax.stop_gradient(hidden_u_new),
                        jax.lax.stop_gradient(hidden_a_new),
                        jax.lax.stop_gradient(out_u_new),
                        jnp.array(0.0),
                        new_total_loss,
                        total_loss_ll + (loss if self.label_last else 0.0),
                        num_spikes,
                        new_params,
                        new_opt_state,
                    )

                should_update = jnp.logical_or(
                    jnp.logical_and(
                        jnp.logical_and(t % self.tbptt_steps == 0, loss_val > 1e-7),
                        train,
                    ),
                    jnp.logical_and(
                        jnp.logical_and(t == sequence_length - 1, loss_val > 1e-7),
                        train,
                    )
                )

                return jax.lax.cond(
                    should_update,
                    apply_grads,
                    lambda: (
                        hidden_z_new,
                        hidden_u_new,
                        hidden_a_new,
                        out_u_new,
                        new_loss_val,
                        new_total_loss,
                        total_loss_ll + (loss if self.label_last else 0.0),
                        num_spikes,
                        params,
                        opt_state,
                    )
                )

            def no_update():
                return (
                    hidden_z_new,
                    hidden_u_new,
                    hidden_a_new,
                    out_u_new,
                    loss_val,
                    total_loss,
                    total_loss_ll,
                    num_spikes,
                    params,
                    opt_state,
                )

            new_carry = jax.lax.cond(t >= self.sub_seq_length, compute_loss_and_update, no_update)

            out_t = jax.lax.cond(t >= self.sub_seq_length, lambda: out_u_new, lambda: jnp.zeros_like(out_u_new))
            return new_carry, out_t

        init_hidden_z = jnp.zeros((batch_size, self.hidden_size))
        init_hidden_u = jnp.zeros_like(init_hidden_z)
        init_hidden_a = jnp.zeros_like(init_hidden_z)
        init_out_u = jnp.zeros((batch_size, self.output_size))
        init_loss_val = jnp.array(0.0)
        init_total_loss = jnp.array(0.0)
        init_total_loss_ll = jnp.array(0.0)
        init_num_spikes = jnp.array(0.0)

        carry_init = (
            init_hidden_z,
            init_hidden_u,
            init_hidden_a,
            init_out_u,
            init_loss_val,
            init_total_loss,
            init_total_loss_ll,
            init_num_spikes,
            params,
            opt_state,
        )

        carry_final, outputs = jax.lax.scan(step_fn, carry_init, jnp.arange(sequence_length))

        if self.label_last:
            outputs = outputs[-self.n_last:]
            final_loss = carry_final[6]
        else:
            final_loss = carry_final[5]

        return (
            outputs,
            final_loss,
            carry_final[7],  # num_spikes
            carry_final[8],  # new_params
            carry_final[9],  # new_opt_state
        )






