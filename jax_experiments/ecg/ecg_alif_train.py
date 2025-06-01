from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
import scipy
import math
import numpy as np # Added for np.mean
from datetime import datetime
import random
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import flax # Added for serialization if you want to save model
import sys
sys.path.append("../..")

import tools # Assuming this 'tools.py' file contains JAX-compatible functions if used directly, or you've re-implemented them as jax functions.

from jax_snn.models import SimpleALIFRNN

# ---------------------------
# Dataset preparation (same as original, but note on device usage)
# ---------------------------

# In JAX, you typically don't set a global device like in PyTorch.
# JAX operations automatically run on the available accelerator (GPU/TPU) if present.
# The 'device' variable here is still a torch.device, which isn't directly used by JAX.
# We'll just print a general message about JAX devices.
print("JAX devices available:", jax.devices())

# PyTorch DataLoader settings
# pin_memory and num_workers are relevant for PyTorch DataLoaders.
# For JAX, data conversion to JAX arrays will happen after loading.
pin_memory = torch.cuda.is_available()
num_workers = 1 if torch.cuda.is_available() else 0

preprocessed_train_dataset = scipy.io.loadmat('data/QTDB_train.mat')
whole_train_dataset = tools.convert_data_format(preprocessed_train_dataset)

total_train_dataset_size = len(whole_train_dataset)
val_dataset_size = int(total_train_dataset_size * 0.1)
train_dataset_size = total_train_dataset_size - val_dataset_size

train_dataset, val_dataset = random_split(
    dataset=whole_train_dataset,
    lengths=[train_dataset_size, val_dataset_size]
)

preprocessed_test_dataset = scipy.io.loadmat('data/QTDB_test.mat')
test_dataset = tools.convert_data_format(preprocessed_test_dataset)
test_dataset_size = len(test_dataset)

sequence_length = 1300
input_size = 4
hidden_size = 36
num_classes = 6

train_batch_size = 4
val_batch_size = 61 # Note: If this is intended to be the full validation set, it might be large.
test_batch_size = 141 # Note: If this is intended to be the full test set, it might be large.

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
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

# ---------------------------
# Model setup (Flax assumed)
# ---------------------------

mask_prob = 0.0
adaptive_tau_mem_mean = 20.
adaptive_tau_mem_std = .5
adaptive_tau_adp_mean = 7.
adaptive_tau_adp_std = .2
out_adaptive_tau_mem_mean = 20.
out_adaptive_tau_mem_std = .5
sub_seq_length = 10
hidden_bias = True
output_bias = True

# Initialize your flax model here
model = SimpleALIFRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    mask_prob=mask_prob,
    adaptive_tau_mem_mean=adaptive_tau_mem_mean,
    adaptive_tau_mem_std=adaptive_tau_mem_std,
    adaptive_tau_adp_mean=adaptive_tau_adp_mean,
    adaptive_tau_adp_std=adaptive_tau_adp_std,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    sub_seq_length=sub_seq_length,
    hidden_bias=hidden_bias,
    output_bias=output_bias
)

def to_jax(batch):
    inputs, targets = batch
    # Convert to JAX array and move to device in one step
    # Note: .numpy() is crucial to get a NumPy array from a PyTorch tensor
    # before converting to JAX.
    inputs = jax.device_put(jnp.array(inputs.numpy()))
    targets = jax.device_put(jnp.array(targets.numpy()))
    # Permute dimensions to (sequence_length, batch_size, features)
    return jnp.transpose(inputs, (1, 0, 2)), jnp.transpose(targets, (1, 0, 2))

@jax.jit
def nll_loss_fn(logits, labels):
    """Optimized NLL loss calculation."""
    log_probs = jax.nn.log_softmax(logits)
    # labels are expected to be class indices (integers)
    return -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)

@jax.jit
def apply_seq_loss_jax(outputs, targets):
    """Optimized sequence loss calculation.
    Outputs are (sequence_length, batch_size, num_classes)
    Targets are (sequence_length, batch_size, num_classes) (one-hot)
    """
    # Convert one-hot targets to class indices for NLL loss
    targets_indices = jnp.argmax(targets, axis=2)
    # vmap over batch and then over sequence length for per-element loss
    # The inner vmap applies nll_loss_fn to each time step
    # The outer vmap applies the result to each batch in parallel
    losses_per_element = jax.vmap(jax.vmap(nll_loss_fn))(outputs, targets_indices)
    # Sum up all losses and then average by the effective sequence length
    return jnp.sum(losses_per_element) / (outputs.shape[0] * outputs.shape[1]) # total_elements = seq_len * batch_size

@jax.jit
def count_correct_prediction_jax(predictions, targets):
    """Optimized correct prediction count.
    Predictions are (sequence_length, batch_size, num_classes) (logits)
    Targets are (sequence_length, batch_size, num_classes) (one-hot)
    """
    predicted_classes = jnp.argmax(predictions, axis=2)
    true_classes = jnp.argmax(targets, axis=2)
    return jnp.sum(predicted_classes == true_classes)

# ---------------------------
# Training state (optimized)
# ---------------------------

class TrainState(train_state.TrainState):
    batch_stats: Any  # For batch norm if needed (though not typically in SNNs without explicit batch norm layers)
    key: jax.random.PRNGKey  # For stochastic operations like dropout, or in this case, model initialization

# ---------------------------
# Training & evaluation steps (optimized)
# ---------------------------

@partial(jax.jit, static_argnames=('sub_seq_length',))
def train_step(state, batch, sub_seq_length):
    """Optimized training step."""
    inputs, targets = batch
    # Targets are sliced based on sub_seq_length
    targets_sliced = targets[sub_seq_length:, :, :]

    def loss_fn(params):
        # The model's apply_fn needs to be called with the parameters
        outputs, _, _ = state.apply_fn({'params': params}, inputs)
        # Apply loss to the sliced outputs and targets
        loss = apply_seq_loss_jax(outputs, targets_sliced)
        return loss, outputs # Return sliced outputs for accuracy calc

    (loss, outputs_sliced), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    correct = count_correct_prediction_jax(outputs_sliced, targets_sliced)
    # Correct accuracy calculation based on sliced outputs/targets shape
    accuracy = (correct / (outputs_sliced.shape[0] * outputs_sliced.shape[1])) * 100.0
    return state, loss, accuracy

@partial(jax.jit, static_argnames=('sub_seq_length',))
def eval_step(state, batch, sub_seq_length):
    """Optimized evaluation step."""
    inputs, targets = batch
    targets_sliced = targets[sub_seq_length:, :, :]
    outputs, _, _ = state.apply_fn({'params': state.params}, inputs)
    loss = apply_seq_loss_jax( outputs, targets_sliced)
    correct = count_correct_prediction_jax(outputs, targets_sliced)
    accuracy = (correct / (outputs.shape[0] * outputs.shape[1])) * 100.0
    return loss, accuracy

# ---------------------------
# Learning rate scheduling (optimized)
# ---------------------------

optimizer_lr = 0.05
epochs_num = 400 # Define this early as it's needed for the scheduler
# Calculate total_train_steps (should be done before create_optimizer call)
total_train_steps = len(train_loader)

def create_optimizer():
    """Create optimizer with linear decay schedule."""
    lr_schedule = optax.linear_schedule(
        init_value=optimizer_lr,
        end_value=0.0,
        transition_steps=epochs_num * total_train_steps
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=lr_schedule)
    )

# ---------------------------
# Initialize model (optimized)
# ---------------------------

def init_model():
    """Initialize model with proper device placement."""
    rng = jax.random.PRNGKey(42)
    # Dummy input needs to match the expected shape: (sequence_length, batch_size, input_size)
    dummy_input = jnp.ones((sequence_length, train_batch_size, input_size))
    # Initialize the model's parameters and potential batch_stats
    variables = model.init(rng, dummy_input)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=create_optimizer(),
        # Ensure batch_stats is handled correctly if it exists, otherwise provide an empty dict
        batch_stats=variables.get('batch_stats', {}),
        key=rng # Initialize the key for the train state
    )

# ---------------------------
# Training loop (optimized)
# ---------------------------

def train_epoch(state, train_loader):
    """Process one training epoch."""
    losses = []
    accuracies = []

    # Iterate through batches
    for i, batch in enumerate(train_loader):
        batch = to_jax(batch) # Convert PyTorch batch to JAX arrays
        state, loss, acc = train_step(state, batch, sub_seq_length)
        losses.append(loss)
        accuracies.append(acc)

        # commment out for actual runs
        if i % 10 == 0: # Print every 10 batches
            print(f"  Batch {i+1}/{len(train_loader)} | Train Loss: {loss:.4f}, Acc: {acc:.2f}%")

    # Aggregate results for the epoch
    return state, jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))

def evaluate(state, loader, dataset_size):
    """Evaluate on full dataset at once.
    This assumes the entire dataset (or validation/test set) fits in memory after concatenation.
    If not, you'd need to iterate through batches here as well.
    """
    all_inputs, all_targets = [], []
    for batch in loader:
        inputs, targets = to_jax(batch)
        all_inputs.append(inputs)
        all_targets.append(targets)

    # Concatenate all batches along the batch dimension (axis=1)
    inputs_concatenated = jnp.concatenate(all_inputs, axis=1)
    targets_concatenated = jnp.concatenate(all_targets, axis=1)

    loss, acc = eval_step(state, (inputs_concatenated, targets_concatenated), sub_seq_length)
    return loss, acc

def train():
    """Optimized training procedure."""
    state = init_model()
    best_state = state
    min_val_loss = float('inf')
    min_val_epoch = 0 # Track the epoch of the best validation loss

    # Record the precise start time
    training_start_time = datetime.now()
    print(f"Training started at: {training_start_time.strftime('%m-%d_%H-%M-%S')}")
    print(f"Initial model parameters:\n{jax.tree_map(lambda x: x.shape, state.params)}")


    # Initial evaluation before training (Epoch 0)
    val_loss, val_acc = evaluate(state, val_loader, val_dataset_size)
    test_loss, test_acc = evaluate(state, test_loader, test_dataset_size)
    print(f"Epoch {0:3d}/{epochs_num} | Summary | Val Loss: {val_loss:.6f}, Acc: {val_acc:8.4f} | "
          f"Test Loss: {test_loss:.6f}, Acc: {test_acc:8.4f}")

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_state = state
        min_val_epoch = 0
        # Optional: Save initial best model if it's the best so far
        # save_model_params(best_state.params, f"models/{start_time}_init_best_model.msgpack")


    for epoch in range(1, epochs_num + 1): # Start from epoch 1 for training loop
        # Training
        state, train_loss, train_acc = train_epoch(state, train_loader)
        print(f"Epoch {epoch:4d}/{epochs_num} | Loss/train: {train_loss:.6f}, Accuracy/train: {train_acc:8.4f}")


        # Validation and Test (after training for the current epoch)
        val_loss, val_acc = evaluate(state, val_loader, val_dataset_size)
        test_loss, test_acc = evaluate(state, test_loader, test_dataset_size)

        print(f"Epoch {epoch:4d}/{epochs_num} | Summary | Loss/val: {val_loss:.6f}, Accuracy/val: {val_acc:8.4f} | "
              f"Loss/test: {test_loss:.6f}, Acc: {test_acc:8.4f}")

        # Check for best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_state = state # Save the state of the best performing model
            min_val_epoch = epoch
            # Optional: Save the best model during training
            # save_model_params(best_state.params, f"models/{start_time}_best_model.msgpack")

        # Check for NaN loss (though JAX tends to be more stable)
        if jnp.isnan(train_loss) or jnp.isnan(val_loss):
            print("NaN loss detected. Ending training.")
            break

    # Record the precise end time
    training_end_time = datetime.now()
    elapsed_time = training_end_time - training_start_time

    print(f'Min val loss: {min_val_loss:.6f} at epoch {min_val_epoch}')
    print(f"Training ended at: {training_end_time.strftime('%m-%d_%H-%M-%S')}")
    print(f"Elapsed Time: {elapsed_time}") # This will print in HH:MM:SS.microseconds format
    return best_state, min_val_loss, min_val_epoch


# Helper function to save model parameters
def save_model_params(params, file_path):
    with open(file_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Model parameters saved to {file_path}")

# Run training
final_state, final_min_val_loss, final_min_val_epoch = train()

# Save your final best model params as needed
# Construct a comment string for the save path, similar to PyTorch version
rand_num = random.randint(1, 10000)
opt_str = "{}_Adam({:.2f}),NLL,LinearLR".format(rand_num, optimizer_lr)
net_str = "RSNN(4,36,6,sub_seq_{},bs_{},ep_{},h_o_bias(True))".format(sub_seq_length, train_batch_size, epochs_num)
unit_str = "ALIF(tau_m({},{}),tau_a({},{}),linMask_{})LI(tau_m({},{}))".format(
    adaptive_tau_mem_mean, adaptive_tau_mem_std, adaptive_tau_adp_mean, adaptive_tau_adp_std, mask_prob,
    out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std)
comment = opt_str + "," + net_str + "," + unit_str

save_path_final = f"models/{datetime.now().strftime('%m-%d_%H-%M-%S')}_{comment}_final.msgpack"
save_path_best_val = f"models/{datetime.now().strftime('%m-%d_%H-%M-%S')}_{comment}_best_val_loss_{final_min_val_loss:.6f}_epoch_{final_min_val_epoch}.msgpack"

# Save the final state's parameters
save_model_params(final_state.params, save_path_final)

print(f"Training complete. Best validation loss: {final_min_val_loss:.6f} at epoch {final_min_val_epoch}")
