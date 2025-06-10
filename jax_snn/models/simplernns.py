from typing import Callable, Any

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
    adaptive_tau_mem: bool = True
    adaptive_tau_mem_mean: float = 20.0
    adaptive_tau_mem_std: float = 5.0
    adaptive_tau_adp: bool = True
    adaptive_tau_adp_mean: float = 150.0
    adaptive_tau_adp_std: float = 50.0
    out_adaptive_tau: bool = True
    out_adaptive_tau_mem_mean: float = 20.0
    out_adaptive_tau_mem_std: float = 5.0
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

        self.scanned_core = nn.scan(
            self.ALIFScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden, self.out)

    def __call__(self, x):
        batch_size = x.shape[1]

        hidden_z = jnp.zeros((batch_size, self.hidden_size))
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_a = jnp.zeros_like(hidden_z)
        out_u = jnp.zeros((batch_size, self.output_size))
        num_spikes = jnp.array(0.0, dtype=jnp.float32)

        init_carry = (hidden_z, hidden_u, hidden_a, out_u, num_spikes)

        (final_hidden_z, final_hidden_u, final_hidden_a, final_out_u, total_spikes), outputs = self.scanned_core(init_carry, x)

        if self.sub_seq_length > 0:
            outputs = outputs[self.sub_seq_length:]

        if self.label_last:
            outputs = outputs[-self.n_last:]

        return outputs, ((final_hidden_z, final_hidden_u, final_hidden_a), final_out_u), total_spikes

    class ALIFScanCore(nn.Module):
        hidden: nn.Module
        out: nn.Module

        @nn.compact
        def __call__(self, carry, input_t):
            hidden_z, hidden_u, hidden_a, out_u, num_spikes = carry

            new_hidden_z, new_hidden_u, new_hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=-1),
                (hidden_z, hidden_u, hidden_a)
            )

            new_out_u = self.out(new_hidden_z, out_u)
            new_num_spikes = num_spikes + jnp.sum(new_hidden_z)

            new_carry = (new_hidden_z, new_hidden_u, new_hidden_a, new_out_u, new_num_spikes)
            return new_carry, new_out_u

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
    n_last: int = 1

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

        self.scanned_core = nn.scan(
            self.DoubleALIFScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden1, self.hidden2, self.out)

    def __call__(self, x):
        batch_size = x.shape[1]

        hidden1_z = jnp.zeros((batch_size, self.hidden1_size))
        hidden1_u = jnp.zeros_like(hidden1_z)
        hidden1_a = jnp.zeros_like(hidden1_z)

        hidden2_z = jnp.zeros((batch_size, self.hidden2_size))
        hidden2_u = jnp.zeros_like(hidden2_z)
        hidden2_a = jnp.zeros_like(hidden2_z)

        out_u = jnp.zeros((batch_size, self.output_size))

        init_carry = (
            hidden1_z, hidden1_u, hidden1_a,
            hidden2_z, hidden2_u, hidden2_a,
            out_u
        )

        (final_hidden1_z, final_hidden1_u, final_hidden1_a,
         final_hidden2_z, final_hidden2_u, final_hidden2_a,
         final_out_u), outputs = self.scanned_core(init_carry, x)

        if self.sub_seq_length > 0:
            outputs = outputs[self.sub_seq_length:]

        if self.label_last:
            outputs = outputs[-self.n_last:]

        return outputs, ((final_hidden1_z, final_hidden1_u, final_hidden1_a,
                          final_hidden2_z, final_hidden2_u, final_hidden2_a), final_out_u)

    class DoubleALIFScanCore(nn.Module):
        hidden1: nn.Module
        hidden2: nn.Module
        out: nn.Module

        @nn.compact
        def __call__(self, carry, input_t):
            hidden1_z, hidden1_u, hidden1_a, hidden2_z, hidden2_u, hidden2_a, out_u = carry

            new_hidden1_z, new_hidden1_u, new_hidden1_a = self.hidden1(
                jnp.concatenate((input_t, hidden1_z), axis=-1),
                (hidden1_z, hidden1_u, hidden1_a)
            )

            new_hidden2_z, new_hidden2_u, new_hidden2_a = self.hidden2(
                jnp.concatenate((new_hidden1_z, hidden2_z), axis=-1),
                (hidden2_z, hidden2_u, hidden2_a)
            )

            new_out_u = self.out(new_hidden2_z, out_u)

            new_carry = (new_hidden1_z, new_hidden1_u, new_hidden1_a,
                         new_hidden2_z, new_hidden2_u, new_hidden2_a,
                         new_out_u)
            return new_carry, new_out_u

class ALIFRSNN_SD(SimpleALIFRNN):
    spike_del_p: float = 0.

    def setup(self):
        super().setup()

        self.scanned_core = nn.scan(
            self.ALIFRSNN_SD_ScanCore,
            variable_broadcast="params",
            split_rngs={"params": True, "spike_deletion": True},
            in_axes=0,
            out_axes=0
        )(self.hidden, self.out)

    def __call__(self, x):
        batch_size = x.shape[1]

        hidden_z = jnp.zeros((batch_size, self.hidden_size))
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_a = jnp.zeros_like(hidden_z)
        out_u = jnp.zeros((batch_size, self.output_size))
        num_spikes = jnp.array(0.0, dtype=jnp.float32)

        init_carry = (hidden_z, hidden_u, hidden_a, out_u, num_spikes)

        (final_hidden_z, final_hidden_u, final_hidden_a, final_out_u, total_spikes), outputs = self.scanned_core(init_carry, x)

        if self.sub_seq_length > 0:
            outputs = outputs[self.sub_seq_length:]

        if self.label_last:
            outputs = outputs[-self.n_last:]

        return outputs, ((final_hidden_z, final_hidden_u, final_hidden_a), final_out_u), total_spikes

    class ALIFRSNN_SD_ScanCore(nn.Module):
        hidden: nn.Module
        out: nn.Module

        @nn.compact
        def __call__(self, carry, input_t):
            hidden_z, hidden_u, hidden_a, out_u, num_spikes = carry

            new_hidden_z, new_hidden_u, new_hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=-1),
                (hidden_z, hidden_u, hidden_a)
            )


            spike_del_p_from_parent = self.parent.spike_del_p
            new_hidden_z = spike_deletion(
                hidden_z=new_hidden_z,
                spike_del_p=spike_del_p_from_parent,
                key=self.make_rng('spike_deletion'),
            )

            new_num_spikes = num_spikes + jnp.sum(new_hidden_z)
            new_out_u = self.out(new_hidden_z, out_u)

            new_carry = (new_hidden_z, new_hidden_u, new_hidden_a, new_out_u, new_num_spikes)
            return new_carry, new_out_u

class ALIFRSNN_BP(SimpleALIFRNN):
    bit_precision: int = 8

    def setup(self):
        super().setup()


        self.scanned_core = nn.scan(
            self.ALIFRSNN_BP_ScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden, self.out)

    def __call__(self, x):
        batch_size = x.shape[1]

        x = quantize_tensor(x, self.bit_precision)

        hidden_z = quantize_tensor(jnp.zeros((batch_size, self.hidden_size)), self.bit_precision)
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_a = jnp.zeros_like(hidden_z)
        out_u = quantize_tensor(jnp.zeros((batch_size, self.output_size)), self.bit_precision)

        num_spikes = jnp.array(0.0, dtype=jnp.float32)

        init_carry = (hidden_z, hidden_u, hidden_a, out_u, num_spikes)

        (final_hidden_z, final_hidden_u, final_hidden_a, final_out_u, total_spikes), outputs = self.scanned_core(init_carry, x)

        if self.sub_seq_length > 0:
            outputs = outputs[self.sub_seq_length:]

        if self.label_last:
            outputs = outputs[-self.n_last:]

        return outputs, ((final_hidden_z, final_hidden_u, final_hidden_a), final_out_u), total_spikes

    class ALIFRSNN_BP_ScanCore(nn.Module):
        hidden: nn.Module
        out: nn.Module

        @nn.compact
        def __call__(self, carry, input_t):
            hidden_z, hidden_u, hidden_a, out_u, num_spikes = carry

            new_hidden_z, new_hidden_u, new_hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=-1),
                (hidden_z, hidden_u, hidden_a)
            )

            bit_precision_from_parent = self.parent.bit_precision
            new_hidden_z = quantize_tensor(new_hidden_z, bit_precision_from_parent)

            new_num_spikes = num_spikes + jnp.sum(new_hidden_z)
            new_out_u = self.out(new_hidden_z, out_u)
            new_out_u = quantize_tensor(new_out_u, bit_precision_from_parent)

            new_carry = (new_hidden_z, new_hidden_u, new_hidden_a, new_out_u, new_num_spikes)
            return new_carry, new_out_u



@jax.jit
def nll_loss_tbptt(logits: jnp.ndarray, labels: jnp.ndarray, num_classes=10) -> jnp.ndarray:
    """Negative log likelihood loss returning scalar mean over batch."""
    log_probs = jax.nn.log_softmax(logits)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    per_example_loss = -jnp.sum(log_probs * one_hot_labels, axis=-1)
    return per_example_loss.mean()


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

    class TbpttScanCoreEval(nn.Module):
        hidden: nn.Module
        out: nn.Module

        @nn.compact
        def __call__(self, carry, ALL_input_t):
            input_t, target_t = ALL_input_t
            hidden_z, hidden_u, hidden_a, out_u, num_spikes, total_loss, idx = carry

            new_hidden_z, new_hidden_u, new_hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=-1),
                (hidden_z, hidden_u, hidden_a)
            )

            new_num_spikes = num_spikes + jnp.sum(new_hidden_z)
            new_out_u = self.out(new_hidden_z, out_u)

            def add_loss(total_loss):
                return total_loss + nll_loss_tbptt(logits=new_out_u, labels=target_t)

            def no_loss(total_loss):
                return total_loss

            new_total_loss = jax.lax.cond(
                idx >= self.sub_seq_length,
                add_loss,
                no_loss,
                operand=total_loss
            )

            new_carry = (
                new_hidden_z,
                new_hidden_u,
                new_hidden_a,
                new_out_u,
                new_num_spikes,
                new_total_loss,
                idx + 1
            )
            return new_carry, new_out_u




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

        self.scanned_core = nn.scan(
            self.TbpttScanCoreEval,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden, self.out)


    def __call__(self, x, y):
        batch_size = x.shape[1]

        hidden_z = jnp.zeros((batch_size, self.hidden_size))
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_a = jnp.zeros_like(hidden_z)
        out_u = jnp.zeros((batch_size, self.output_size))


        init_carry = (hidden_z, hidden_u, hidden_a, out_u, jnp.array(0.0), 0.0, 0)

        (final_hidden_z, final_hidden_u, final_hidden_a, final_out_u, total_spikes, final_total_loss, _), outputs = self.scanned_core(
            init_carry, (x,y))

        outputs = outputs[self.sub_seq_length:]

        if self.label_last:
            outputs = outputs[-self.n_last:, : , :]
            final_total_loss = 0.0

        return outputs, final_total_loss, total_spikes










