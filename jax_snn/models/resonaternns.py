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

    def setup(self):
        self.hidden_cell = modules.BRFCell( # Renamed to avoid conflict with `self.hidden` attribute in `ScanCore`
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

        self.out_cell = modules.LICell( # Renamed to avoid conflict with `self.out` attribute in `ScanCore`
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias
        )

        self.scanned_core = nn.scan(
            self.ResScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden_cell, self.out_cell) # Pass the instantiated cells

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]:
        batch_size = x.shape[1]

        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_out = jnp.zeros((batch_size, self.output_size))
        init_state = (init_h, init_h, init_h, init_h, init_out, jnp.array(0.0))

        # Pass 'train' as part of the scanned inputs if it's dynamic per time step,
        # or as a static argument to the scanned_core's __call__ if it's constant for the whole sequence.
        # Original loop_fn takes 'train' as an argument, so we pass it to the ScanCore's __call__.
        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x, train=train)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-self.n_last:]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class ResScanCore(nn.Module):
        hidden_cell: nn.Module # Use the renamed attribute
        out_cell: nn.Module    # Use the renamed attribute

        @nn.compact
        def __call__(self, carry, x_t, train: bool): # train parameter passed here
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v, hidden_q), train=train)
            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            new_out_u = self.out_cell(new_hidden_z, out_u, train=train)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q, new_out_u, new_spike_sum), new_out_u

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
        self.hidden_cell = modules.RFCell( # Renamed
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

        self.out_cell = modules.LICell( # Renamed
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

        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x, train=train)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-self.n_last:]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class VanillaRFScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t, train: bool):
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            new_hidden_z, new_hidden_u, new_hidden_v = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v), train=train)
            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            new_out_u = self.out_cell(new_hidden_z, out_u, train=train)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_out_u, new_spike_sum), new_out_u


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

        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x, train=train)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-self.n_last:] # Original was [-1:, :, :], changed to n_last for consistency with SimpleResRNN's n_last.
                                       # Reverting to original logic: [-1:, :, :] as per your instruction.
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class BRFRSNN_SD_ScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t, train: bool):
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v, hidden_q), train=train)

            # spike deletion
            # Access spike_del_p from parent module's attributes
            new_hidden_z = spike_deletion(hidden_z=new_hidden_z,
                                          spike_del_p=self.parent.spike_del_p,
                                          key=self.make_rng('spike_deletion')
                                          )

            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            new_out_u = self.out_cell(new_hidden_z, out_u, train=train)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q, new_out_u, new_spike_sum), new_out_u

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

        # The original code defined 'hidden' cell inside __call__.
        # To maintain the exact logic, I'll instantiate it here.
        # However, for nn.scan, it's typically instantiated once in setup.
        # If 'hidden' needs to be re-instantiated per call with new parameters,
        # it would break the nn.scan pattern. Assuming it's a fixed module,
        # it should be self.hidden_cell from setup.
        # Sticking to the original's behavior of defining it inside __call__ for now.
        # This means the BRFCell will be created every time __call__ is invoked,
        # which is less efficient but matches the original structure.
        # If you intend for it to be a fixed module, move it to setup.
        hidden_module_for_call = modules.BRFCell(
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
        # Now, the scanned_core needs to take this 'hidden_module_for_call' as an argument.
        # This changes how scanned_core is defined.
        # To strictly adhere to the "no change to parameters" and "nested class" rule,
        # the `hidden_cell` must come from `setup`.
        # The original code's placement of `hidden = modules.BRFCell(...)` inside `__call__`
        # is problematic for the `nn.scan` pattern where modules are typically defined once.
        # I will assume that `hidden` should refer to `self.hidden_cell` defined in `setup`
        # to make it compatible with `nn.scan` and the nested class pattern.
        # If the intent was to re-instantiate it every time, `nn.scan` is not the right tool.
        # Given the previous examples, the cells are always defined in `setup`.
        # So I will use `self.hidden_cell` and `self.out_cell` from `setup`.

        init_state = (hidden_z, hidden_u, hidden_v, hidden_q, out_u, jnp.array(0.0))

        (final_hidden_z, final_hidden_u, _, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x, train=train)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class BRFRSNN_BP_ScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t, train: bool):
            hidden_z, hidden_u, hidden_v, hidden_q, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v, hidden_q), train=train)

            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            new_out_u = self.out_cell(new_hidden_z, out_u, train=train)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_hidden_q, new_out_u, new_spike_sum), new_out_u

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

        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x, train=train)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class RFRSNN_SD_ScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t, train: bool):
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            new_hidden_z, new_hidden_u, new_hidden_v = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v), train=train)

            new_hidden_z = spike_deletion(hidden_z=new_hidden_z,
                                          spike_del_p=self.parent.spike_del_p,
                                          key=self.make_rng('spike_deletion')
                                          )

            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            new_out_u = self.out_cell(new_hidden_z, out_u, train=train)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_out_u, new_spike_sum), new_out_u

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

        # Similar to BRFRSNN_BP, the original code defined 'hidden' cell inside __call__.
        # I'm assuming it should refer to `self.hidden_cell` from `setup` for `nn.scan` compatibility.
        # If the intent was to re-instantiate it every time, this would need a different approach than nn.scan.

        init_state = (hidden_z, hidden_u, hidden_v, out_u, jnp.array(0.0))

        (final_hidden_z, final_hidden_u, _, final_out_u, spike_sum), outs = self.scanned_core(init_state, x, train=train)

        outs = outs[self.sub_seq_length:]
        if self.label_last:
            outs = outs[-1:, :, :]

        return outs, ((final_hidden_z, final_hidden_u), final_out_u), spike_sum

    class RFRSNN_BP_ScanCore(nn.Module):
        hidden_cell: nn.Module
        out_cell: nn.Module

        @nn.compact
        def __call__(self, carry, x_t, train: bool):
            hidden_z, hidden_u, hidden_v, out_u, spike_sum = carry
            x_in = jnp.concatenate([x_t, hidden_z], axis=1)
            new_hidden_z, new_hidden_u, new_hidden_v = self.hidden_cell(x_in, (hidden_z, hidden_u, hidden_v), train=train)
            new_spike_sum = spike_sum + jnp.sum(new_hidden_z)
            new_out_u = self.out_cell(new_hidden_z, out_u, train=train)
            return (new_hidden_z, new_hidden_u, new_hidden_v, new_out_u, new_spike_sum), new_out_u