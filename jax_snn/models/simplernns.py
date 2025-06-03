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

        # Scan-compatible wrapper
        self.scanned_core = nn.scan(
            self.ALIFScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden, self.out)

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[1]

        hidden_z = jnp.zeros((batch_size, self.hidden_size))
        hidden_u = jnp.zeros_like(hidden_z)
        hidden_a = jnp.zeros_like(hidden_z)
        out_u = jnp.zeros((batch_size, self.output_size))
        num_spikes = jnp.array(0.0, dtype=jnp.float32)

        init_carry = (hidden_z, hidden_u, hidden_a, out_u, num_spikes)

        # Apply scan core
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
    n_last: int = 1 # Added for consistency with SimpleALIFRNN, though not used in original logic

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

    @nn.compact
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
            outputs = outputs[-self.n_last:] # Changed to n_last for consistency

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
        # Call super setup to initialize hidden and out modules
        super().setup()

        # Re-define scanned_core to use the ALIFRSNN_SD_ScanCore
        self.scanned_core = nn.scan(
            self.ALIFRSNN_SD_ScanCore,
            variable_broadcast="params",
            split_rngs={"params": True, "spike_deletion": True}, # Split rng for spike_deletion
            in_axes=0,
            out_axes=0
        )(self.hidden, self.out)

    @nn.compact
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
        # spike_del_p will be broadcasted from the parent module

        @nn.compact
        def __call__(self, carry, input_t):
            hidden_z, hidden_u, hidden_a, out_u, num_spikes = carry

            new_hidden_z, new_hidden_u, new_hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=-1),
                (hidden_z, hidden_u, hidden_a)
            )

            # Access spike_del_p from parent module's attributes
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
    bit_precision: int = 8 # Assuming a default bit precision

    def setup(self):
        # Call super setup to initialize hidden and out modules
        super().setup()

        # Re-define scanned_core to use the ALIFRSNN_BP_ScanCore
        self.scanned_core = nn.scan(
            self.ALIFRSNN_BP_ScanCore,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(self.hidden, self.out)

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[1]

        # Quantize input before passing to scan
        x = quantize_tensor(x, self.bit_precision)

        hidden_z = quantize_tensor(jnp.zeros((batch_size, self.hidden_size)), self.bit_precision)
        hidden_u = jnp.zeros_like(hidden_z) # u is not quantized in original
        hidden_a = jnp.zeros_like(hidden_z) # a is not quantized in original
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
        # bit_precision will be broadcasted from the parent module

        @nn.compact
        def __call__(self, carry, input_t):
            hidden_z, hidden_u, hidden_a, out_u, num_spikes = carry

            new_hidden_z, new_hidden_u, new_hidden_a = self.hidden(
                jnp.concatenate((input_t, hidden_z), axis=-1),
                (hidden_z, hidden_u, hidden_a)
            )

            # Access bit_precision from parent module's attributes
            bit_precision_from_parent = self.parent.bit_precision
            new_hidden_z = quantize_tensor(new_hidden_z, bit_precision_from_parent)

            new_num_spikes = num_spikes + jnp.sum(new_hidden_z)
            new_out_u = self.out(new_hidden_z, out_u)
            new_out_u = quantize_tensor(new_out_u, bit_precision_from_parent)

            new_carry = (new_hidden_z, new_hidden_u, new_hidden_a, new_out_u, new_num_spikes)
            return new_carry, new_out_u




class SimpleALIFRNNTbptt(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int
    criterion: Callable # Expects a JAX-compatible loss function

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

    # Define the RNN step function as a nested class within the parent module.
    # This allows it to access the parent's sub-modules (hidden_cell, out_cell).
    class _RNNScanStep(nn.Module):
        parent_module: 'SimpleALIFRNNTbptt' # Reference to the outer module instance
        sequence_length: int
        train: bool
        optimizer: Any

        @nn.compact
        def __call__(self, carry, t_idx):
            # Unpack the carry state
            hidden_z, hidden_u, hidden_a, out_u, loss_val, total_loss, total_loss_ll, num_spikes, params, opt_state = carry

            # Extract inputs and targets for the current time step
            # Note: x and y are implicitly captured from the outer __call__ scope
            # (or could be passed as part of the scan's `xs` if preferred, but current setup works)
            input_t = self.parent_module.x_data[t_idx]
            target_t = self.parent_module.y_data[t_idx]

            # --- Forward Pass ---
            # Concatenate input and previous hidden spikes for the recurrent cell
            hidden_input = jnp.concatenate([input_t, hidden_z], axis=1)

            # Apply the hidden ALIF cell. Parameters are accessed via the 'params' dict.
            # We assume 'params' contains a sub-dict for 'hidden_cell' and 'out_cell'.
            # 'mutable=False' means no internal mutable states (like batch stats) are updated here.
            # 'rngs=None' means no random number generation (e.g., dropout) is used.
            hidden_z_new, (hidden_u_new, hidden_a_new) = self.parent_module.hidden_cell.apply(
                {'params': params['hidden_cell']}, # Access specific sub-module parameters
                hidden_input,
                (hidden_z, hidden_u, hidden_a),
                mutable=False,
                rngs=None
            )

            # Apply the output LI cell
            out_u_new = self.parent_module.out_cell.apply(
                {'params': params['out_cell']}, # Access specific sub-module parameters
                hidden_z_new,
                out_u,
                mutable=False,
                rngs=None
            )

            # Accumulate total spikes
            spikes = jnp.sum(hidden_z_new)
            num_spikes += spikes

            # --- Loss Computation and Parameter Update Logic (TBPTT) ---
            def compute_loss_and_update_branch():
                # Determine if log_softmax is needed based on the criterion
                if hasattr(self.parent_module.criterion, '__name__') and self.parent_module.criterion.__name__ == 'nll_loss':
                    logits = nn.log_softmax(out_u_new, axis=1)
                else:
                    logits = out_u_new

                # Compute loss for the current step
                loss = self.parent_module.criterion(logits, target_t)
                new_loss_val = loss_val + loss # Accumulate loss for the TBPTT segment
                new_total_loss = total_loss + loss # Accumulate total loss over the entire sequence

                def apply_grads_branch():
                    # Define the loss function for jax.grad.
                    # It takes the current parameters and computes the loss for this step.
                    # Crucially, it uses the *previous* hidden states (from 'carry') as inputs
                    # to correctly compute gradients for the current step's computation.
                    def loss_fn(current_params_for_grad):
                        h_z_grad, (h_u_grad, h_a_grad) = self.parent_module.hidden_cell.apply(
                            {'params': current_params_for_grad['hidden_cell']},
                            hidden_input, # Use hidden_input from the current step
                            (hidden_z, hidden_u, hidden_a), # Use carry states from the beginning of the step
                            mutable=False,
                            rngs=None
                        )
                        o_u_grad = self.parent_module.out_cell.apply(
                            {'params': current_params_for_grad['out_cell']},
                            h_z_grad,
                            out_u, # Use carry state from the beginning of the step
                            mutable=False,
                            rngs=None
                        )
                        # Reapply log_softmax if needed for the loss function
                        lgt = nn.log_softmax(o_u_grad, axis=1) if hasattr(self.parent_module.criterion, '__name__') and self.parent_module.criterion.__name__ == 'nll_loss' else o_u_grad
                        return self.parent_module.criterion(lgt, target_t)

                    # Compute gradients
                    grads = jax.grad(loss_fn)(params)

                    # Apply updates using the optimizer
                    updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                    new_params = optax.apply_updates(params, updates)

                    # Return updated states.
                    # For TBPTT, we stop gradients for the hidden states after an update.
                    # This is equivalent to .detach_() in PyTorch.
                    return (
                        jax.lax.stop_gradient(hidden_z_new), # Detach
                        jax.lax.stop_gradient(hidden_u_new), # Detach
                        jax.lax.stop_gradient(hidden_a_new), # Detach
                        jax.lax.stop_gradient(out_u_new),   # Detach
                        jnp.array(0.0), # Reset accumulated loss for the next segment
                        new_total_loss,
                        total_loss_ll + (loss if self.parent_module.label_last else 0.0),
                        num_spikes,
                        new_params,
                        new_opt_state,
                    )

                # Condition to decide if gradients should be applied
                # This mimics the PyTorch logic: update every tbptt_steps or at the very end
                should_update = jnp.logical_or(
                    jnp.logical_and(
                        jnp.logical_and(t_idx % self.parent_module.tbptt_steps == 0, loss_val > 1e-7),
                        self.train, # Only update if in training mode
                    ),
                    jnp.logical_and(
                        jnp.logical_and(t_idx == self.sequence_length - 1, loss_val > 1e-7),
                        self.train, # Only update if in training mode
                    )
                )

                # Conditionally apply gradients or just continue accumulating loss
                return jax.lax.cond(
                    should_update,
                    apply_grads_branch,
                    lambda: (
                        hidden_z_new,
                        hidden_u_new,
                        hidden_a_new,
                        out_u_new,
                        new_loss_val, # Continue accumulating loss
                        new_total_loss,
                        total_loss_ll + (loss if self.parent_module.label_last else 0.0),
                        num_spikes,
                        params, # Parameters remain unchanged if no update
                        opt_state, # Optimizer state remains unchanged if no update
                    )
                )

            def no_loss_update_branch():
                # This branch is taken if t_idx < self.sub_seq_length.
                # No loss is computed or accumulated, and no updates happen.
                return (
                    hidden_z_new,
                    hidden_u_new,
                    hidden_a_new,
                    out_u_new,
                    loss_val, # Loss value remains as is (not accumulating yet)
                    total_loss, # Total loss remains as is
                    total_loss_ll, # Total loss for label_last remains as is
                    num_spikes,
                    params,
                    opt_state,
                )

            # Decide whether to compute loss and update based on sub_seq_length
            new_carry = jax.lax.cond(
                t_idx >= self.parent_module.sub_seq_length,
                compute_loss_and_update_branch,
                no_loss_update_branch
            )

            # Output for this time step. Only collect if t_idx >= sub_seq_length.
            out_t = jax.lax.cond(
                t_idx >= self.parent_module.sub_seq_length,
                lambda: out_u_new,
                lambda: jnp.zeros_like(out_u_new) # Return zeros if not collecting output yet
            )
            return new_carry, out_t

    def setup(self):
        # Initialize the recurrent cell and output cell
        self.hidden_cell = modules.ALIFCell(
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

        self.out_cell = modules.LICell(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem=self.out_adaptive_tau,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bias=self.output_bias,
        )

    def __call__(self, x, y, params, opt_state, optimizer, train: bool = True):
        sequence_length, batch_size = x.shape[0], x.shape[1]

        # Store x and y as temporary attributes to be accessed by the nested _RNNScanStep.
        # This is a common pattern when the scan body needs access to inputs that are not part of `xs`.
        self.x_data = x
        self.y_data = y

        # Initialize the carry state for the scan
        init_hidden_z = jnp.zeros((batch_size, self.hidden_size))
        init_hidden_u = jnp.zeros_like(init_hidden_z)
        init_hidden_a = jnp.zeros_like(init_hidden_z)
        init_out_u = jnp.zeros((batch_size, self.output_size))
        init_loss_val = jnp.array(0.0) # Accumulated loss for the current TBPTT segment
        init_total_loss = jnp.array(0.0) # Total loss over the entire sequence
        init_total_loss_ll = jnp.array(0.0) # Total loss for label_last
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
            params, # Parameters are part of the carry because they are updated
            opt_state, # Optimizer state is part of the carry because it is updated
        )

        # Instantiate the nested _RNNScanStep module.
        # Pass 'self' (the SimpleALIFRNNTbptt instance) to its constructor
        # so it can access the parent's attributes and sub-modules.
        rnn_scan_step_module = self._RNNScanStep(
            parent_module=self,
            sequence_length=sequence_length,
            train=train,
            optimizer=optimizer
        )

        # Perform the scan operation.
        # The `rnn_scan_step_module` is the callable that defines each step.
        # `variable_broadcast='params'` ensures parameters are available at each step.
        # `split_rngs={'params': False}` indicates no separate RNG keys are needed for params.
        # `length=sequence_length` specifies the number of steps.
        # `unroll=1` is crucial for TBPTT, ensuring gradients are truncated at each step.
        carry_final, outputs = nn.scan(
            rnn_scan_step_module,
            variable_broadcast='params',
            split_rngs={'params': False},
            length=sequence_length,
            reverse=False,
            unroll=1
        )(carry_init, jnp.arange(sequence_length)) # The second argument to scan is `xs` (iterated over)

        # Clean up temporary attributes
        del self.x_data
        del self.y_data

        # Final loss calculation based on label_last
        if self.label_last:
            # Ensure outputs has enough elements to slice
            outputs = outputs[-self.n_last:]
            final_loss = carry_final[6] # total_loss_ll
        else:
            final_loss = carry_final[5] # total_loss

        return (
            outputs,
            final_loss,
            carry_final[7],  # num_spikes
            carry_final[8],  # new_params (final updated parameters)
            carry_final[9],  # new_opt_state (final updated optimizer state)
        )




