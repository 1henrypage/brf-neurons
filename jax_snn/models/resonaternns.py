import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
from .. import modules
from ..functional import spike_deletion, quantize_tensor

# --- SimpleResRNN ---
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

    def setup(self):
        self.hidden_cell = modules.BRFCell(
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

        self.out_cell = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

        # The nn.scan is defined directly in setup
        self.scanned_core = nn.scan(
            self.ResScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden_cell, self.out_cell)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_h, init_out, jnp.array(0.0))

        # Pass 'train' as a static argument to the scanned_core's __call__
        # The 'train' argument here is *not* for the internal cells but for the `nn.scan` itself
        # if the scanned function needed to differentiate train/eval specific behaviors.
        # However, since the cells themselves no longer take 'train', this 'train' parameter
        # to scanned_core might become unused unless there's logic *within* ResScanCore
        # that depends on it (e.g., if it had dropout layers directly).
        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x) # Removed train=train from here

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-self.n_last:]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class ResScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t): # Removed train: bool from here
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            # Call hidden_cell and out_cell without the 'train' argument
            new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v, hidden_q))
            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            new_out_u = self.out_cell(new_hidden_z, out_u)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q, new_out_u, new_spike_sum), new_out_u

# SimpleVanillaRFRNN

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

    def setup(self):
        self.hidden_cell = modules.RFCell(
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

        self.out_cell = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

        self.scanned_core = nn.scan(
            self.VanillaRFScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden_cell, self.out_cell)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_out, jnp.array(0.0))

        # Removed train=train from here
        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-self.n_last:]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class VanillaRFScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t): # Removed train: bool from here
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            # Call hidden_cell and out_cell without the 'train' argument
            new_hidden_z, new_hidden_u, new_hidden_v = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v))
            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            new_out_u = self.out_cell(new_hidden_z, out_u)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_out_u, new_spike_sum), new_out_u


# BRFRSNN_SD

class BRFRSNN_SD(SimpleResRNN):
    spike_del_p: float = 0.0

    def setup(self):
        super().setup() # Calls SimpleResRNN's setup to initialize hidden_cell and out_cell

        # Re-define scanned_core to use the BRFRSNN_SD_ScanCore
        self.scanned_core = nn.scan(
            self.BRFRSNN_SD_ScanCore,
            variable_broadcast="params",
            split_rngs={"params": False, "spike_deletion": True}, # Split rng for spike_deletion
            in_axes=0,
            out_axes=0
        )(self.hidden_cell, self.out_cell) # Pass the already initialized cells

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_h, init_out, jnp.array(0.0))

        # Removed train=train from here
        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :] # Original was [-1:, :, :], changed to n_last for consistency with SimpleResRNN's n_last.
                                   # Reverting to original logic: [-1:, :, :] as per your instruction.

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class BRFRSNN_SD_ScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t): # Removed train: bool from here
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            # Call hidden_cell without the 'train' argument
            new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v, hidden_q))

            # spike deletion
            # Access spike_del_p from parent module's attributes
            new_hidden_z = spike_deletion(hidden_z=new_hidden_z,
                                          spike_del_p=self.parent.spike_del_p,
                                          key=self.make_rng('spike_deletion')
                                          )

            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            # Call out_cell without the 'train' argument
            new_out_u = self.out_cell(new_hidden_z, out_u)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q, new_out_u, new_spike_sum), new_out_u


# BRFRSNN_BP

class BRFRSNN_BP(SimpleResRNN):
    bit_precision: int = 52

    def setup(self):
        # Call SimpleResRNN's setup first to initialize hidden_cell
        super().setup()
        # Then override self.out_cell with LICellBP
        self.out_cell = modules.LICellBP(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            bit_precision=self.bit_precision,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std
        )

        # Re-define scanned_core to use the BRFRSNN_BP_ScanCore
        self.scanned_core = nn.scan(
            self.BRFRSNN_BP_ScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden_cell, self.out_cell) # Pass the initialized cells

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        x = quantize_tensor(x, self.bit_precision)

        hidden_z = quantize_tensor(jnp.zeros((batch_size, self.hidden_size)), self.bit_precision)
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_v = jnp.zeros_like(hidden_z)
        hidden_q = jnp.zeros_like(hidden_z)

        out_u = quantize_tensor(jnp.zeros((batch_size, self.output_size)), self.bit_precision)

        init_state = (hidden_z, hidden_u, hidden_v, hidden_q, out_u, jnp.array(0.0))

        # Removed train=train from here
        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class BRFRSNN_BP_ScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t): # Removed train: bool from here
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            # Call hidden_cell without the 'train' argument
            new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v, hidden_q))

            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            # Call out_cell without the 'train' argument
            new_out_u = self.out_cell(new_hidden_z, out_u)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q, new_out_u, new_spike_sum), new_out_u


# RFRSNN_SD

class RFRSNN_SD(SimpleVanillaRFRNN):
    spike_del_p: float = 0.0

    def setup(self):
        super().setup() # Calls SimpleVanillaRFRNN's setup to initialize hidden_cell and out_cell

        # Re-define scanned_core to use the RFRSNN_SD_ScanCore
        self.scanned_core = nn.scan(
            self.RFRSNN_SD_ScanCore,
            variable_broadcast="params",
            split_rngs={"params": False, "spike_deletion": True}, # Split rng for spike_deletion
            in_axes=0,
            out_axes=0
        )(self.hidden_cell, self.out_cell) # Pass the already initialized cells

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_out, jnp.array(0.0))

        # Removed train=train from here
        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class RFRSNN_SD_ScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t): # Removed train: bool from here
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            # Call hidden_cell without the 'train' argument
            new_hidden_z, new_hidden_u, new_hidden_v = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v))

            new_hidden_z = spike_deletion(hidden_z=new_hidden_z,
                                          spike_del_p=self.parent.spike_del_p,
                                          key=self.make_rng('spike_deletion')
                                          )

            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            # Call out_cell without the 'train' argument
            new_out_u = self.out_cell(new_hidden_z, out_u)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_out_u, new_spike_sum), new_out_u


# RFRSNN_BP

class RFRSNN_BP(SimpleVanillaRFRNN):
    bit_precision: int = 32

    def setup(self):
        super().setup() # Calls SimpleVanillaRFRNN's setup to initialize hidden_cell
        # Then override self.out_cell with LICellBP
        self.out_cell = modules.LICellBP(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            bit_precision=self.bit_precision,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std
        )

        # Re-define scanned_core to use the RFRSNN_BP_ScanCore
        self.scanned_core = nn.scan(
            self.RFRSNN_BP_ScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden_cell, self.out_cell) # Pass the initialized cells

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        x = quantize_tensor(x, self.bit_precision)
        hidden_z = quantize_tensor(jnp.zeros((batch_size, self.hidden_size)), self.bit_precision)
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_v = jnp.zeros_like(hidden_z)
        out_u = quantize_tensor(jnp.zeros((batch_size, self.output_size)), self.bit_precision)

        init_state = (hidden_z, hidden_u, hidden_v, out_u, jnp.array(0.0))

        # Removed train=train from here
        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class RFRSNN_BP_ScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t): # Removed train: bool from here
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            # Call hidden_cell without the 'train' argument
            new_hidden_z, new_hidden_u, new_hidden_v = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v))
            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            # Call out_cell without the 'train' argument
            new_out_u = self.out_cell(new_hidden_z, out_u)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_out_u, new_spike_sum), new_out_u