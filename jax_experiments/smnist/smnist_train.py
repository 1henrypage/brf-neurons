from functools import partial
import sys
from typing import Any, Callable, Tuple
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime
import random
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import flax
import sys
sys.path.append("../..")
import time
from torch.utils.tensorboard import SummaryWriter
import subprocess
import os
import math

# -------------------------------------------------------------------
# Device Configuration
# -------------------------------------------------------------------

device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
print("Using device:", device)

# -------------------------------------------------------------------
# Dataset Preparation (PSMNIST specific)
# -------------------------------------------------------------------

PERMUTED = True
label_last = False
sequence_length = 28 * 28
input_size = 1
num_classes = 10
batch_size = 256
val_batch_size = 256
test_batch_size = 256

start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
rand_num = random.randint(1, 10000)

# Create permutation index
if PERMUTED:
    permuted_idx = jax.random.permutation(jax.random.PRNGKey(42), sequence_length)
else:
    permuted_idx = jnp.arange(sequence_length)

# Save permutation index
os.makedirs("models", exist_ok=True)
np.save(f"models/{start_time}_{rand_num}_permuted_idx.npy", np.array(permuted_idx))

# Load datasets
train_dataset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

total_dataset_size = len(train_dataset)
val_dataset_size = int(total_dataset_size * 0.1)
train_dataset_size = total_dataset_size - val_dataset_size

train_dataset, val_dataset = random_split(
    train_dataset, [train_dataset_size, val_dataset_size]
)

test_dataset = torchvision.datasets.MNIST(
    root="data",
    train=False,
    transform=torchvision.transforms.ToTensor()
)

test_dataset_size = len(test_dataset)

# -------------------------------------------------------------------
# DataLoader Configuration
# -------------------------------------------------------------------

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    drop_last=False
)


# -------------------------------------------------------------------
# JAX Utilities
# -------------------------------------------------------------------

def smnist_transform_input_batch(tensor: torch.Tensor, permuted_idx: jnp.ndarray) -> jnp.ndarray:
    """Transform input batch to permuted sequence format."""
    # Convert to JAX array and reshape
    array = jnp.array(tensor.numpy())
    array = array.reshape(array.shape[0], sequence_length, input_size)
    array = array.transpose(1, 0, 2)  # [seq_len, batch, features]

    # Apply permutation
    return array[permuted_idx]


def to_jax(batch: Tuple[torch.Tensor, torch.Tensor], permuted_idx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert PyTorch batch to JAX arrays with permutation."""
    inputs, targets = batch
    inputs_jax = smnist_transform_input_batch(inputs, permuted_idx)
    targets_jax = jax.device_put(jnp.array(targets.numpy()), device=device)
    return inputs_jax, targets_jax


@jax.jit
def nll_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Negative log likelihood loss."""
    log_probs = jax.nn.log_softmax(logits)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    return -jnp.sum(log_probs * one_hot_labels, axis=-1)


@partial(jax.jit, static_argnames=('label_last',))
def apply_seq_loss(
        outputs: jnp.ndarray,
        targets: jnp.ndarray,
        label_last: bool
) -> jnp.ndarray:
    """Compute sequence loss."""
    if label_last:
        # Only use last output
        loss = nll_loss(outputs[-1], targets)
    else:
        # Use all outputs
        loss = jnp.mean(nll_loss(outputs, targets), axis=0)

    return jnp.mean(loss)


@jax.jit
def count_correct(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Count correct predictions by comparing mean outputs."""
    preds = jnp.argmax(jnp.mean(outputs, axis=0), axis=-1)
    return jnp.sum(preds == targets)


# -------------------------------------------------------------------
# Model Configuration
# -------------------------------------------------------------------

from jax_snn.models import SimpleResRNN, SimpleVanillaRFRNN

hidden_size = 256
mask_prob = 0.0
omega_a = 15.
omega_b = 85.
b_offset_a = 0.1
b_offset_b = 1.
out_adaptive_tau_mem_mean = 20.
out_adaptive_tau_mem_std = 1.
output_bias = False

model = SimpleVanillaRFRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    adaptive_omega_a=omega_a,
    adaptive_omega_b=omega_b,
    adaptive_b_offset_a=b_offset_a,
    adaptive_b_offset_b=b_offset_b,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    output_bias=output_bias,
)


# -------------------------------------------------------------------
# Training State
# -------------------------------------------------------------------

class TrainState(train_state.TrainState):
    batch_stats: Any
    key: jax.random.PRNGKey


# -------------------------------------------------------------------
# Training & Evaluation Steps
# -------------------------------------------------------------------

@partial(jax.jit, static_argnames=('label_last',))
def train_step(
        state: TrainState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        label_last: bool
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """Single training step."""

    def loss_fn(params):
        outputs, _, _ = state.apply_fn(
            {'params': params},
            inputs,
            rngs={'dropout': jax.random.fold_in(state.key, state.step)}
        )
        loss = apply_seq_loss(outputs, targets, label_last)
        return loss, outputs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    correct = count_correct(outputs, targets)
    return state, loss, correct


@partial(jax.jit, static_argnames=('label_last',))
def eval_step(
        state: TrainState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        label_last: bool
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluation step."""
    outputs, _, spikes = state.apply_fn(
        {'params': state.params},
        inputs
    )
    loss = apply_seq_loss(outputs, targets, label_last)
    correct = count_correct(outputs, targets)
    return loss, correct, jnp.sum(spikes)


# -------------------------------------------------------------------
# Learning Rate Scheduling
# -------------------------------------------------------------------

optimizer_lr = 0.1
epochs_num = 300
total_train_steps = len(train_loader) * epochs_num
gradient_clip_value = 1.0


def create_optimizer():
    """Create optimizer with linear decay and gradient clipping."""
    lr_schedule = optax.linear_schedule(
        init_value=optimizer_lr,
        end_value=0.0,
        transition_steps=total_train_steps
    )

    # Gradient clipping
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip_value),
        optax.adam(learning_rate=lr_schedule)
    )


# -------------------------------------------------------------------
# Model Initialization
# -------------------------------------------------------------------

def init_model(rng: jax.random.PRNGKey) -> TrainState:
    """Initialize model and training state."""
    dummy_input = jnp.ones((sequence_length, batch_size, input_size))
    variables = model.init(rng, dummy_input)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=create_optimizer(),
        batch_stats=variables.get('batch_stats', {}),
        key=rng
    )


# -------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------

def train():
    # Initialize model and state
    rng = jax.random.PRNGKey(42)
    state = init_model(rng)
    best_params = state.params
    min_val_loss = float('inf')
    min_val_epoch = 0
    end_training = False

    # TensorBoard setup
    opt_str = f"{rand_num}_Adam({optimizer_lr}),NLL,script-bw,LinLR,LL({label_last}),PERMUTED({PERMUTED})"
    net_str = f"1,{hidden_size},10,bs={batch_size},ep={epochs_num}"
    unit_str = f"BRF_omega{omega_a}_{omega_b}b{b_offset_a}_{b_offset_b},LI{out_adaptive_tau_mem_mean}_{out_adaptive_tau_mem_std}"
    comment = opt_str + "," + net_str + "," + unit_str

    writer = SummaryWriter(comment=comment)
    log_dir = writer.log_dir

    # Start TensorBoard
    port = 6006
    while True:
        try:
            subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", str(port)])
            break
        except OSError:
            port += 1

    start_time_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    save_path = f"models/{start_time_str}_{comment}.msgpack"
    save_init_path = f"models/{start_time_str}_init_{comment}.msgpack"

    # Save initial parameters
    with open(save_init_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(state.params))

    print(f"{start_time_str} {comment}")

    # Training loop
    iteration = 0
    for epoch in range(epochs_num + 1):
        # Validation
        val_loss, val_correct = 0.0, 0
        for batch in val_loader:
            inputs, targets = to_jax(batch, permuted_idx)
            loss, correct, _ = eval_step(state, inputs, targets, label_last)
            val_loss += loss.item() * targets.shape[0]
            val_correct += correct.item()

        val_loss /= len(val_dataset)
        val_acc = 100.0 * val_correct / len(val_dataset)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Test
        test_loss, test_correct, test_spikes = 0.0, 0, 0
        for batch in test_loader:
            inputs, targets = to_jax(batch, permuted_idx)
            loss, correct, spikes = eval_step(state, inputs, targets, label_last)
            test_loss += loss.item() * targets.shape[0]
            test_correct += correct.item()
            test_spikes += spikes.item()

        test_loss /= len(test_dataset)
        test_acc = 100.0 * test_correct / len(test_dataset)
        test_sop = test_spikes / len(test_dataset)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("SOP/test", test_sop, epoch)

        print(f"Epoch {epoch:4d}/{epochs_num} | Val Loss: {val_loss:.6f}, Acc: {val_acc:.2f}% | "
              f"Test Loss: {test_loss:.6f}, Acc: {test_acc:.2f}% | SOP: {test_sop:.2f}")

        # Save best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            best_params = state.params
            with open(save_path, 'wb') as f:
                f.write(flax.serialization.to_bytes(best_params))
            print(f"Saved new best model with val loss: {val_loss:.6f}")

        # Training epoch
        if epoch < epochs_num and not end_training:
            state = state.replace(key=jax.random.split(state.key)[0])
            epoch_loss, epoch_correct = 0.0, 0
            epoch_start = time.time()
            batch_count = 0
            print_train_loss = 0.0
            print_correct = 0
            print_total = 0

            for i, batch in enumerate(train_loader):
                inputs, targets = to_jax(batch, permuted_idx)
                state, loss, correct = train_step(state, inputs, targets, label_last)

                # Accumulate metrics
                current_batch_size = targets.shape[0]
                epoch_loss += loss.item() * current_batch_size
                epoch_correct += correct.item()
                print_train_loss += loss.item()
                print_total += current_batch_size
                print_correct += correct.item()

                iteration += 1
                batch_count += 1

                # Log training metrics
                writer.add_scalar("Loss/train", loss.item(), iteration)
                writer.add_scalar("Accuracy/train", 100.0 * correct.item() / current_batch_size, iteration)

                # Print batch statistics
                if (i + 1) % 150 == 0:
                    print_acc = 100.0 * print_correct / print_total
                    print(f"Epoch {epoch + 1:4d}/{epochs_num} | Step {i + 1:4d}/{len(train_loader)} | "
                          f"Loss: {print_train_loss / 150:.6f}, Acc: {print_acc:.2f}%")
                    print_train_loss = 0.0
                    print_correct = 0
                    print_total = 0

                # Check for NaN
                if jnp.isnan(loss).any():
                    print("NaN loss detected, stopping training")
                    end_training = True
                    break

            # Epoch statistics
            epoch_loss /= len(train_dataset)
            epoch_acc = 100.0 * epoch_correct / len(train_dataset)
            epoch_time = time.time() - epoch_start
            writer.add_scalar("Time/epoch", epoch_time, epoch)
            print(f"Epoch {epoch + 1:4d}/{epochs_num} | Train Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")

    writer.close()
    print(f"Min val loss: {min_val_loss:.6f} at epoch {min_val_epoch}")
    return best_params


# Run training
if __name__ == "__main__":
    best_params = train()
