from functools import partial
from typing import Any, Callable
import torch
from torch.utils.data import DataLoader, random_split
import math
import numpy as np
from datetime import datetime
import random
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import flax
import sys
import time
import subprocess
from dataclasses import field # Import field

# Ensure necessary paths are appended for custom modules like `tools` and `jax_snn.models`
sys.path.append("../..")

import tools
from jax_snn.models import SimpleResRNN  # Assuming this is your JAX equivalent
from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------------------------
## Global Configuration & Dataset Preparation
# -------------------------------------------------------------------

print("JAX devices available:", jax.devices())

# Determine pin_memory and num_workers based on CUDA availability
pin_memory = torch.cuda.is_available()
num_workers = 1 if torch.cuda.is_available() else 0

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Load and split datasets
whole_train_dataset = tools.shd_to_dataset('data/trainX_sample.npy', 'data/trainY_sample.npy') # Changed to sample data as in PyTorch
total_train_dataset_size = len(whole_train_dataset)
val_dataset_size = int(total_train_dataset_size * 0.1)
train_dataset_size = total_train_dataset_size - val_dataset_size
train_dataset, val_dataset = random_split(
    dataset=whole_train_dataset,
    lengths=[train_dataset_size, val_dataset_size]
)

test_dataset = tools.shd_to_dataset('data/testX_sample.npy', 'data/testY_sample.npy') # Changed to sample data as in PyTorch
test_dataset_size = len(test_dataset)

# Global model and training parameters
sequence_length = 250
input_size = 700
hidden_size = 128
num_classes = 20
batch_size = 32

val_batch_size = 256
test_batch_size = 256

# DataLoader setup
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False,
    drop_last=False
)

# -------------------------------------------------------------------
# Model Definition (Flax)
# -------------------------------------------------------------------

delta = 0.01 #4e-3
mask_prob = 0.0
omega_a = 5.
omega_b = 10.
b_offset_a = 2.
b_offset_b = 3.
out_adaptive_tau_mem_mean = 20.
out_adaptive_tau_mem_std = 5.

label_last = False
sub_seq_length = 0 # This needs to be 0 for full sequence processing, as in PyTorch

hidden_bias = False # This isn't used in the model definition in the original snippet, keeping it for clarity
output_bias = False

# Initialize the JAX/Flax model
model = SimpleResRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    adaptive_omega_a=omega_a,
    adaptive_omega_b=omega_b,
    adaptive_b_offset_a=b_offset_a,
    adaptive_b_offset_b=b_offset_b,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    label_last=label_last, # This is passed to the model's internal logic
    sub_seq_length=sub_seq_length, # This is passed to the model's internal logic
    mask_prob=mask_prob,
    output_bias=output_bias, # Make sure this matches if you have output_bias in PyTorch
    dt=delta
)

# Helper function to convert PyTorch DataLoader batches to JAX arrays and transpose inputs
def to_jax(batch):
    inputs, targets = batch
    # Convert to NumPy first, then to JAX array for CPU/GPU placement
    inputs = jax.device_put(jnp.array(inputs.numpy()))
    targets = jax.device_put(jnp.array(targets.numpy()))
    # Permute inputs from (batch_size, sequence_length, data_size) to (sequence_length, batch_size, data_size)
    # to match the model's expected input shape.
    return jnp.transpose(inputs, (1, 0, 2)), targets

# -------------------------------------------------------------------
## Loss and Metrics Functions
# -------------------------------------------------------------------

@jax.jit
def nll_loss_fn(log_probs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """JAX version of NLLLoss"""
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    return -jnp.sum(log_probs * one_hot_labels, axis=-1)

@partial(jax.jit, static_argnames=('label_last', 'sub_seq_length'))
def apply_seq_loss_jax(
        outputs: jnp.ndarray,
        targets: jnp.ndarray,  # INTEGER class labels (batch_size,)
        label_last: bool,
        sub_seq_length: int
) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(outputs, axis=-1)  # shape: [T, B, C]

    if label_last:
        # Only use last timestep
        return jnp.mean(nll_loss_fn(log_probs[-1], targets))
    else:
        # Use all timesteps starting from sub_seq_length
        valid_log_probs = log_probs[sub_seq_length:]  # shape: [T', B, C]

        # Vectorize over time
        batched_loss = jax.vmap(nll_loss_fn, in_axes=(0, None))(valid_log_probs, targets)  # shape: [T', B]
        mean_loss_per_step = jnp.mean(batched_loss, axis=1)  # [T']
        return jnp.mean(mean_loss_per_step)  # scalar


# Corrected evaluation metrics
@jax.jit
def count_correct_predictions_jax(log_probs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Count correct predictions from LOG probabilities"""
    pred_classes = jnp.argmax(log_probs, axis=-1)
    return jnp.sum(pred_classes == targets)


# -------------------------------------------------------------------
## Flax Training State
# -------------------------------------------------------------------

class TrainState(train_state.TrainState):
    batch_stats: Any = None
    key: jax.random.PRNGKey = field(default_factory=lambda: jax.random.PRNGKey(42))

# -------------------------------------------------------------------
## Learning Rate Scheduling & Optimizer
# -------------------------------------------------------------------

optimizer_lr = 0.075
epochs_num = 20
total_steps = len(train_loader) # Number of batches per epoch

def create_optimizer(epochs: int, total_steps_per_epoch: int, initial_lr: float) -> optax.GradientTransformation:
    """
    Creates an Optax optimizer with a linear learning rate schedule.
    Args:
        epochs (int): Total number of training epochs.
        total_steps_per_epoch (int): Number of batches per epoch.
        initial_lr (float): Initial learning rate.
    Returns:
        optax.GradientTransformation: The configured Optax optimizer.
    """
    lr_schedule = optax.linear_schedule(
        init_value=initial_lr,
        end_value=0.0,
        transition_steps=epochs * total_steps_per_epoch # Total steps over all epochs
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Equivalent to gradient_clip_value
        optax.adam(learning_rate=lr_schedule)
    )

# -------------------------------------------------------------------
## Model Initialization
# -------------------------------------------------------------------

def init_model() -> TrainState:
    """
    Initializes the Flax model and creates the initial TrainState.
    Returns:
        TrainState: The initial training state.
    """
    rng = jax.random.PRNGKey(42) # This rng is used for model initialization
    # Dummy input for model initialization to infer parameter shapes
    # (sequence_length, batch_size, input_size)
    dummy_input = jnp.ones((sequence_length, batch_size, input_size))
    variables = model.init(rng, dummy_input) # `init` returns a dict with 'params' and potentially 'batch_stats'

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=create_optimizer(epochs_num, total_steps, optimizer_lr),
        batch_stats=variables.get('batch_stats', None), # Safely get batch_stats if present
        key=rng # Pass the initial key to the TrainState
    )

# -------------------------------------------------------------------
## JIT-compiled Training & Evaluation Steps
# -------------------------------------------------------------------

# Updated train/eval steps
@partial(jax.jit, static_argnames=('label_last', 'sub_seq_length'))
def train_step(state, batch, label_last, sub_seq_length):
    inputs, targets = batch
    targets = targets.astype(jnp.int32)  # Ensure integer labels

    def loss_fn(params):
        outputs, _, _ = state.apply_fn({'params': params}, inputs)
        log_probs = jax.nn.log_softmax(outputs, axis=-1)
        loss = apply_seq_loss_jax(outputs, targets, label_last, sub_seq_length)

        # Accuracy calculation
        if label_last:
            pred_log_probs = log_probs[-1]
        else:
            pred_log_probs = jnp.mean(log_probs, axis=0)

        correct = count_correct_predictions_jax(pred_log_probs, targets)
        return loss, (outputs, correct)

    (loss, (outputs, correct)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, correct

@partial(jax.jit, static_argnames=('label_last', 'sub_seq_length'))
def eval_step(state, batch, label_last, sub_seq_length):
    inputs, targets = batch
    targets = targets.astype(jnp.int32)  # Ensure integer labels

    outputs, _, num_spikes = state.apply_fn({'params': state.params}, inputs)
    log_probs = jax.nn.log_softmax(outputs, axis=-1)
    loss = apply_seq_loss_jax(outputs, targets, label_last, sub_seq_length)

    # Accuracy calculation
    if label_last:
        pred_log_probs = log_probs[-1]
    else:
        pred_log_probs = jnp.mean(log_probs, axis=0)

    correct = count_correct_predictions_jax(pred_log_probs, targets)
    return loss, correct, num_spikes


# -------------------------------------------------------------------
## Training Loop Functions
# -------------------------------------------------------------------

def train_epoch(
        state: TrainState,
        train_loader: DataLoader,
        writer: SummaryWriter,
        epoch_idx: int,
        print_every: int,
        label_last: bool,
        sub_seq_length: int,
        sequence_length: int
) -> tuple[TrainState, bool]:
    """
    Runs a single training epoch.
    Args:
        state (TrainState): Current training state.
        train_loader (DataLoader): DataLoader for the training set.
        writer (SummaryWriter): TensorBoard writer.
        epoch_idx (int): Current epoch number.
        print_every (int): Frequency for printing training progress.
        label_last (bool): Flag for loss scaling.
        sub_seq_length (int): Sub-sequence length for loss scaling.
        sequence_length (int): Full sequence length for loss scaling.
    Returns:
        tuple: (updated_state, end_training_flag)
    """
    print_train_loss_accumulator = 0.0 # Accumulator for scaled loss for printing
    print_correct = 0
    print_total = 0
    end_training_flag = False

    epoch_start_time = time.time()

    for i, batch in enumerate(train_loader):
        jax_inputs, jax_targets = to_jax(batch) # Unpack original batch to get Python batch_size

        state, raw_summed_loss_jax_array, correct_count_jax_array, batch_outputs_mean = train_step(
            state, (jax_inputs, jax_targets), label_last, sub_seq_length
        )

        # --- Convert JAX arrays to Python scalars and apply scaling for logging ---
        # Get concrete raw summed loss
        raw_summed_loss_item = raw_summed_loss_jax_array.item()

        # Apply the label_last scaling logic here, mirroring your PyTorch shd_train.py
        if label_last:
            loss_value_for_logging = raw_summed_loss_item
        else:
            denominator = sequence_length - sub_seq_length
            # Ensure denominator is not zero to avoid division by zero errors
            loss_value_for_logging = raw_summed_loss_item / (denominator if denominator != 0 else 1.0)
        # --- End of JAX to Python conversion and scaling ---

        writer.add_scalar("Loss/train", loss_value_for_logging, epoch_idx * len(train_loader) + i)

        # Calculate accuracy for logging based on batch_outputs_mean and correct_count_jax_array
        current_batch_size = jax_targets.shape[0] # Get current batch size from JAX target
        acc_for_logging = (correct_count_jax_array.item() / current_batch_size) * 100.0
        writer.add_scalar("Accuracy/train", acc_for_logging, epoch_idx * len(train_loader) + i)

        print_train_loss_accumulator += loss_value_for_logging # Use the scaled loss for printout
        print_total += current_batch_size
        print_correct += correct_count_jax_array.item() # Get concrete value for printing

        if math.isnan(loss_value_for_logging):
            end_training_flag = True
            break

        if (i + 1) % print_every == 0:
            print_acc = (print_correct / print_total) * 100.0
            print(
                f"Epoch [{epoch_idx + 1:4d}/{epochs_num:4d}] | Step [{i + 1:4d}/{total_steps:4d}] | "
                f"Loss/train: {print_train_loss_accumulator / print_every:.6f}, Accuracy/train: {print_acc:8.4f}",
                flush=True
            )
            print_correct = 0
            print_total = 0
            print_train_loss_accumulator = 0.0 # Reset for next interval

    epoch_end_time = time.time()
    writer.add_scalar("Time/train_epoch_step", (epoch_end_time - epoch_start_time), epoch_idx)
    writer.flush() # Flush after each epoch

    return state, end_training_flag


def evaluate(
        state: TrainState,
        loader: DataLoader,
        dataset_size: int,
        writer: SummaryWriter,
        epoch: int,
        prefix: str,
        label_last: bool,
        sub_seq_length: int,
        sequence_length: int
) -> tuple[float, float, float]:
    """
    Evaluates the model on a given dataset.
    Args:
        state (TrainState): Current training state.
        loader (DataLoader): DataLoader for the evaluation set.
        dataset_size (int): Total number of samples in the dataset.
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch number.
        prefix (str): Prefix for TensorBoard logs (e.g., "val", "test").
        label_last (bool): Flag for loss scaling.
        sub_seq_length (int): Sub-sequence length for loss scaling.
        sequence_length (int): Full sequence length for loss scaling.
    Returns:
        tuple: (average_loss, average_accuracy, average_spikes_per_sample)
    """
    total_scaled_loss = 0.0 # Accumulate scaled loss for averaging
    total_correct = 0
    total_spikes = 0
    total_batches = 0 # Track number of batches processed for averaging loss

    for i, batch in enumerate(loader):
        inputs, targets = to_jax(batch)

        raw_summed_loss_jax_array, correct_count_jax_array, num_spikes_jax_array = eval_step(
            state, (inputs, targets), label_last, sub_seq_length
        )

        # --- Convert JAX arrays to Python scalars and apply scaling for logging ---
        raw_summed_loss_item = raw_summed_loss_jax_array.item()

        # Apply the label_last scaling logic for logging
        if label_last:
            loss_value_for_logging = raw_summed_loss_item
        else:
            denominator = sequence_length - sub_seq_length
            loss_value_for_logging = raw_summed_loss_item / (denominator if denominator != 0 else 1.0)
        # --- End of JAX to Python conversion and scaling ---

        total_scaled_loss += loss_value_for_logging
        total_correct += correct_count_jax_array.item() # Get concrete value
        total_spikes += num_spikes_jax_array.item() # Get concrete value
        total_batches += 1

    avg_loss = total_scaled_loss / total_batches if total_batches > 0 else 0.0 # Average the scaled loss over batches
    avg_accuracy = (total_correct / dataset_size) * 100.0
    avg_spikes_per_sample = total_spikes / dataset_size # SOP

    writer.add_scalar(f"Loss/{prefix}", avg_loss, epoch)
    writer.add_scalar(f"Accuracy/{prefix}", avg_accuracy, epoch)
    if prefix == "test": # Log SOP only for test set as in original
        writer.add_scalar("SOP/test", avg_spikes_per_sample, epoch)

    return avg_loss, avg_accuracy, avg_spikes_per_sample

# Helper function to save model parameters
def save_model_params(params: Any, file_path: str):
    """
    Saves Flax model parameters to a file using Flax's serialization.
    Args:
        params (Any): The model parameters (state.params).
        file_path (str): Path to save the parameters.
    """
    with open(file_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Model parameters saved to {file_path}")

# -------------------------------------------------------------------
## Main Training Function
# -------------------------------------------------------------------

def train():
    """
    Main training loop for the JAX/Flax model.
    Handles model initialization, training, evaluation, and logging.
    Returns:
        tuple: (best_state_params, min_val_loss, min_val_epoch)
    """
    state = init_model()
    best_state_params = state.params # Initialize with initial params
    min_val_loss = float("inf")
    min_val_epoch = 0

    # TensorBoard setup
    rand_num = random.randint(1, 10000)
    opt_str = f"{rand_num}_Adam({optimizer_lr:.2f}),NLL,script-fgiDG,LinLR,LL({label_last},no_gc)"
    net_str = f"700,{hidden_size},20,bs_{batch_size},ep_{epochs_num}"
    unit_str = f"BRF(omega{omega_a},{omega_b}b{b_offset_a},{b_offset_b})LI({out_adaptive_tau_mem_mean},{out_adaptive_tau_mem_std})"
    comment = f"{opt_str},{net_str},{unit_str}"

    writer = SummaryWriter(comment=comment)
    log_dir = writer.log_dir
    print(f"TensorBoard log directory: {log_dir}")
    # Start TensorBoard process in the background
    try:
        subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6009"])
    except Exception as e:
        print(f"Failed to start TensorBoard process: {e}. You might need to start it manually.")


    start_time_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    print(f"[{start_time_str}] {comment}")

    # Save initial parameters
    save_model_params(state.params, f"models/{start_time_str}_{comment}_init.msgpack")
    print(f"Initial model parameters shape tree:\n{jax.tree_map(lambda x: x.shape, state.params)}")

    # Record the start time of the entire training run
    training_start_time = time.time()

    # Initial evaluation before training (Epoch 0)
    # The PyTorch code runs evaluation for epoch 0 and then proceeds to epoch 0 training.
    # We will mirror this by calling evaluate for epoch 0 before the main training loop.
    val_loss, val_acc, _ = evaluate(state, val_loader, val_dataset_size, writer, 0, "val", label_last, sub_seq_length, sequence_length)
    test_loss, test_acc, test_sop = evaluate(state, test_loader, test_dataset_size, writer, 0, "test", label_last, sub_seq_length, sequence_length)

    print(
        f"Epoch {0:4d}/{epochs_num:4d} | Summary | Loss/val: {val_loss:.6f}, Accuracy/val: {val_acc:.4f}% | "
        f"Loss/test: {test_loss:.6f}, Accuracy/test: {test_acc:.4f}% | SOP: {test_sop:.4f}", flush=True
    )
    writer.flush()

    # Save best model after initial evaluation if it's the best so far
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_state_params = state.params
        min_val_epoch = 0
        save_model_params(best_state_params, f"models/{start_time_str}_{comment}_best_val_loss_{min_val_loss:.6f}.msgpack")

    print_every = 230 # From original Torch code

    # Main training loop
    end_training = False
    # Loop from epoch 0 up to epochs_num (inclusive for the final eval)
    for epoch in range(epochs_num + 1):
        # Only perform training for epochs 0 to epochs_num - 1
        if epoch < epochs_num:
            state, end_training = train_epoch(state, train_loader, writer, epoch, print_every, label_last, sub_seq_length, sequence_length)
            writer.flush() # Ensure all training logs are written

        # Evaluate after each training epoch (and for the final summary at epochs_num)
        val_loss, val_acc, _ = evaluate(state, val_loader, val_dataset_size, writer, epoch, "val", label_last, sub_seq_length, sequence_length)
        test_loss, test_acc, test_sop = evaluate(state, test_loader, test_dataset_size, writer, epoch, "test", label_last, sub_seq_length, sequence_length)

        print(
            f"Epoch [{epoch:4d}/{epochs_num:4d}] | Summary | Loss/val: {val_loss:.6f}, Accuracy/val: {val_acc:.4f}% | "
            f"Loss/test: {test_loss:.6f}, Accuracy/test: {test_acc:.4f}% | SOP: {test_sop:.4f}", flush=True
        )

        # Update best model if current validation loss is lower
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            best_state_params = state.params
            save_model_params(best_state_params, f"models/{start_time_str}_{comment}_best_val_loss_{min_val_loss:.6f}.msgpack")

        writer.flush() # Ensure all evaluation logs are written

        # Break condition if NaN loss is detected during training or evaluation
        if end_training or math.isnan(val_loss): # Use math.isnan for Python float
            print("NaN loss detected or training flagged to end. Ending training prematurely.")
            break

    writer.close()

    total_training_time = time.time() - training_start_time
    print(f"Total training runtime: {total_training_time:.2f} seconds")
    print(f"Minimum val loss: {min_val_loss:.6f} at epoch: {min_val_epoch}")

    return best_state_params, min_val_loss, min_val_epoch

# Run the training process
if __name__ == "__main__":
    final_best_params, final_min_val_loss, final_min_val_epoch = train()
