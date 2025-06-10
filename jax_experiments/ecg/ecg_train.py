from functools import partial
from typing import Any
import torch
from torch.utils.data import DataLoader, random_split
import scipy
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

sys.path.append("../..")

import tools
from jax_snn.models import SimpleResRNN, \
    SimpleVanillaRFRNN  # Assuming jax_snn.models.SimpleResRNN is the JAX equivalent
from torch.utils.tensorboard import SummaryWriter


# -------------------------------------------------------------------
# Dataset Preparation
# -------------------------------------------------------------------

print("JAX devices available:", jax.devices())

pin_memory = torch.cuda.is_available()
num_workers = 1 if torch.cuda.is_available() else 0

preprocessed_train_dataset = scipy.io.loadmat('./data/QTDB_train.mat')
whole_train_dataset = tools.convert_data_format(preprocessed_train_dataset)

total_train_dataset_size = len(whole_train_dataset)
val_dataset_size = int(total_train_dataset_size * 0.1)
train_dataset_size = total_train_dataset_size - val_dataset_size

train_dataset, val_dataset = random_split(
    dataset=whole_train_dataset,
    lengths=[train_dataset_size, val_dataset_size]
)

preprocessed_test_dataset = scipy.io.loadmat('./data/QTDB_test.mat')
test_dataset = tools.convert_data_format(preprocessed_test_dataset)
test_dataset_size = len(test_dataset)

sequence_length = 1300
input_size = 4
num_classes = 6

train_batch_size = 16
val_batch_size = 61
test_batch_size = 141

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

# -------------------------------------------------------------------
# Model Setup (Flax)
# -------------------------------------------------------------------

hidden_size = 36
mask_prob = 0.0 # Not used in SimpleResRNN in original code, but kept for consistency
omega_mean = 3.
omega_std = 5.
b_offset_a = 0.1
b_offset_b = 1.0
out_adaptive_tau_mem_mean = 20.
out_adaptive_tau_mem_std = 1.
sub_seq_length = 0
dt = 0.01

model = SimpleResRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    adaptive_omega_a=omega_mean,
    adaptive_omega_b=omega_std,
    adaptive_b_offset_a=b_offset_a,
    adaptive_b_offset_b=b_offset_b,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    sub_seq_length=sub_seq_length,
    output_bias=False,
)

def to_jax(batch):
    inputs, targets = batch
    inputs = jax.device_put(jnp.array(inputs.numpy()))
    targets = jax.device_put(jnp.array(targets.numpy()))
    return jnp.transpose(inputs, (1, 0, 2)), jnp.transpose(targets, (1, 0, 2))

@jax.jit
def nll_loss_fn(logits, labels):
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)

@jax.jit
def apply_seq_loss_jax(outputs, targets):
    targets_indices = jnp.argmax(targets, axis=2)
    # This vmap is crucial to apply the loss function correctly across sequence length and batch
    losses_per_element = jax.vmap(jax.vmap(nll_loss_fn))(outputs, targets_indices)
    return jnp.sum(losses_per_element) / (outputs.shape[0] * outputs.shape[1])

@jax.jit
def count_correct_prediction_jax(predictions, targets):
    predicted_classes = jnp.argmax(predictions, axis=2)
    true_classes = jnp.argmax(targets, axis=2)
    return jnp.sum(predicted_classes == true_classes)

# -------------------------------------------------------------------
# Training State
# -------------------------------------------------------------------

class TrainState(train_state.TrainState):
    batch_stats: Any
    key: jax.random.PRNGKey

# -------------------------------------------------------------------
# Training & Evaluation Steps
# -------------------------------------------------------------------

@partial(jax.jit, static_argnames=('sub_seq_length',))
def train_step(state, batch, sub_seq_length):
    inputs, targets = batch
    targets_sliced = targets[sub_seq_length:, :, :]

    def loss_fn(params):
        outputs, _, _ = state.apply_fn({'params': params}, inputs)
        loss = apply_seq_loss_jax(outputs, targets_sliced)
        return loss, outputs

    (loss, outputs_sliced), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    correct = count_correct_prediction_jax(outputs_sliced, targets_sliced)
    accuracy = (correct / (outputs_sliced.shape[0] * outputs_sliced.shape[1])) * 100.0
    return state, loss, accuracy, outputs_sliced

@partial(jax.jit, static_argnames=('sub_seq_length',))
def eval_step(state, batch, sub_seq_length):
    inputs, targets = batch
    targets_sliced = targets[sub_seq_length:, :, :]
    outputs, _, num_spikes = state.apply_fn({'params': state.params}, inputs) # Assuming num_spikes is also returned
    loss = apply_seq_loss_jax(outputs, targets_sliced)
    correct = count_correct_prediction_jax(outputs, targets_sliced)
    accuracy = (correct / (outputs.shape[0] * outputs.shape[1])) * 100.0
    return loss, accuracy, jnp.sum(num_spikes) # Return sum of spikes for consistency

# -------------------------------------------------------------------
# Learning Rate Scheduling
# -------------------------------------------------------------------

optimizer_lr = 0.1
epochs_num = 400
total_train_steps = len(train_loader)

def create_optimizer():
    lr_schedule = optax.linear_schedule(
        init_value=optimizer_lr,
        end_value=0.0,
        transition_steps=epochs_num * total_train_steps
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0), # Gradient clipping
        optax.adam(learning_rate=lr_schedule)
    )

# -------------------------------------------------------------------
# Initialize Model
# -------------------------------------------------------------------

def init_model():
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((sequence_length, train_batch_size, input_size))
    variables = model.init(rng, dummy_input)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=create_optimizer(),
        batch_stats=variables.get('batch_stats', {}),
        key=rng
    )

# -------------------------------------------------------------------
# Training Loop Functions
# -------------------------------------------------------------------

def train_epoch(state, train_loader, writer, epoch_idx, total_sequence_elements):
    losses = []
    accuracies = []
    epoch_start_time = time.time()

    for i, batch in enumerate(train_loader):
        batch = to_jax(batch)
        state, loss, acc, outputs_sliced = train_step(state, batch, sub_seq_length)

        losses.append(loss)
        accuracies.append(acc)

        iteration = epoch_idx * len(train_loader) + i
        writer.add_scalar("Loss/train", loss.item(), iteration)
        writer.add_scalar("accuracy/train", acc.item(), iteration)

        # The batch_accuracy calculation in the original PyTorch code for training printout
        # used `current_batch_size * (sequence_length - sub_seq_length)`.
        # For JAX, `outputs_sliced.shape[0]` is sequence_length - sub_seq_length,
        # and `outputs_sliced.shape[1]` is current_batch_size.
        # So it's equivalent to `outputs_sliced.shape[0] * outputs_sliced.shape[1]`
        # already correctly computed as part of `acc` from `train_step`.
        # We can just print the 'acc' directly here.
        if i % 10 == 0:
            print(f"  Batch {i+1}/{len(train_loader)} | Train Loss: {loss:.4f}, Acc: {acc:.2f}%")

        if jnp.isnan(loss):
            print("NaN loss detected during training step. Ending epoch.")
            return state, jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies)), True

    epoch_end_time = time.time()
    writer.add_scalar("Time/train_epoch_step", (epoch_end_time - epoch_start_time), epoch_idx)

    return state, jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies)), False


def evaluate(state, loader, dataset_size, writer, epoch, prefix):
    total_loss = 0
    total_correct = 0
    total_spikes = 0.0 # Placeholder for total spikes, assuming SimpleResRNN also outputs them

    for i, batch in enumerate(loader):
        inputs_jax, targets_jax = to_jax(batch)
        loss, acc, num_spikes = eval_step(state, (inputs_jax, targets_jax), sub_seq_length)
        total_loss += loss.item() * (inputs_jax.shape[1] * (sequence_length - sub_seq_length)) # Reconstruct sum of losses for averaging later
        total_correct += int(acc * (inputs_jax.shape[1] * (sequence_length - sub_seq_length)) / 100.0) # Reconstruct sum of corrects for averaging later
        total_spikes += num_spikes # Accumulate total spikes

    avg_loss = total_loss / (dataset_size * (sequence_length - sub_seq_length))
    avg_acc = (total_correct / (dataset_size * (sequence_length - sub_seq_length))) * 100.0

    writer.add_scalar(f"Loss/{prefix}", avg_loss, epoch)
    writer.add_scalar(f"accuracy/{prefix}", avg_acc, epoch)

    return avg_loss, avg_acc, total_spikes

# Helper function to save model parameters
def save_model_params(params, file_path):
    with open(file_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Model parameters saved to {file_path}")

# -------------------------------------------------------------------
# Main Training Function
# -------------------------------------------------------------------

def train():
    state = init_model()
    best_state_params = state.params # Initialize best_state_params to the initial state's parameters
    min_val_loss = float('inf')
    min_val_epoch = 0

    # TensorBoard setup
    rand_num = random.randint(1, 10000)
    opt_str = "{}_Adam({:.2f}),script-bw,NLL,LinearLR,no_gc".format(rand_num, optimizer_lr)
    net_str = "4,{},6,bs={},ep={}".format(hidden_size, train_batch_size, epochs_num)
    unit_str = "BRF(omega{},{},b{},{})LI({},{})".format(
        omega_mean, omega_std, b_offset_a, b_offset_b, out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std)
    comment = opt_str + "," + net_str + "," + unit_str


    writer = SummaryWriter(comment=comment)
    log_dir = writer.log_dir
    print(f"TensorBoard log directory: {log_dir}")
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])

    start_time_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    print(start_time_str, comment)

    save_model_params(best_state_params, f"models/{start_time_str}_{comment}_init_loss.msgpack")

    training_start_time = datetime.now()
    print(f"Training started at: {training_start_time.strftime('%m-%d_%H-%M-%S')}")
    print(f"Initial model parameters:\n{jax.tree_map(lambda x: x.shape, state.params)}")

    # Initial evaluation before training (Epoch 0)
    val_loss, val_acc, _ = evaluate(state, val_loader, val_dataset_size, writer, 0, "val")
    test_loss, test_acc, test_total_spikes = evaluate(state, test_loader, test_dataset_size, writer, 0, "test")
    print(f"Epoch {0:3d}/{epochs_num} | Summary | Loss/val: {val_loss:.6f}, Accuracy/val: {val_acc:8.4f} | "
          f"Loss/test: {test_loss:.6f}, Accuracy/test: {test_acc:8.4f}")

    writer.flush()

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_state_params = state.params
        min_val_epoch = 0

    total_sequence_elements = (sequence_length - sub_seq_length)
    end_training = False

    for epoch in range(1, epochs_num + 1):
        # Training
        state, train_loss, train_acc, end_training = train_epoch(state, train_loader, writer, epoch - 1, total_sequence_elements)
        print(f"Epoch {epoch:4d}/{epochs_num} | Loss/train: {train_loss:.6f}, Accuracy/train: {train_acc:8.4f}")

        # Validation and Test (after training for the current epoch)
        val_loss, val_acc, _ = evaluate(state, val_loader, val_dataset_size, writer, epoch, "val")
        test_loss, test_acc, test_total_spikes = evaluate(state, test_loader, test_dataset_size, writer, epoch, "test")

        print(f"Epoch {epoch:4d}/{epochs_num} | Summary | Loss/val: {val_loss:.6f}, Accuracy/val: {val_acc:8.4f} | "
              f"Loss/test: {test_loss:.6f}, Acc: {test_acc:8.4f}")

        # Check for best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_state_params = state.params
            min_val_epoch = epoch

        writer.flush()

        # Check for NaN loss
        if end_training or jnp.isnan(val_loss):
            print("NaN loss detected or training flagged to end. Ending training.")
            break

    # Record the precise end time
    training_end_time = datetime.now()
    elapsed_time = training_end_time - training_start_time

    print(f'Min val loss: {min_val_loss:.6f} at epoch {min_val_epoch}')
    print(f"Training ended at: {training_end_time.strftime('%m-%d_%H-%M-%S')}")
    print(f"Elapsed Time: {elapsed_time}")

    # Save the final model (parameters of the best performing model)
    final_save_path = f"models/{start_time_str}_{comment}_final_best_val_loss_{min_val_loss:.6f}_epoch_{min_val_epoch}.msgpack"
    save_model_params(best_state_params, final_save_path)

    writer.close()

    print(f"Training complete. Best validation loss: {min_val_loss:.6f} at epoch {min_val_epoch}")
    return best_state_params, min_val_loss, final_min_val_epoch


# Run training
final_best_params, final_min_val_loss, final_min_val_epoch = train()