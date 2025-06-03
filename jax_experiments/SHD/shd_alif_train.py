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
from jax_snn.models import SimpleALIFRNN
from torch.utils.tensorboard import SummaryWriter


# -------------------------------------------------------------------
# Dataset Preparation
# -------------------------------------------------------------------

print("JAX devices available:", jax.devices())

pin_memory = torch.cuda.is_available()
num_workers = 1 if torch.cuda.is_available() else 0

whole_train_dataset = tools.shd_to_dataset('./data/trainX_4ms.npy', './data/trainY_4ms.npy')

total_train_dataset_size = len(whole_train_dataset)
val_dataset_size = int(total_train_dataset_size * 0.1)
train_dataset_size = total_train_dataset_size - val_dataset_size

train_dataset, val_dataset = random_split(
    dataset=whole_train_dataset,
    lengths=[train_dataset_size, val_dataset_size]
)

test_dataset = tools.shd_to_dataset('./data/testX_4ms.npy', './data/testY_4ms.npy')
test_dataset_size = len(test_dataset)

sequence_length = 250
input_size = 700
hidden_size = 128
num_classes = 20
batch_size = 32

val_batch_size = 256
test_batch_size = 256

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
# Model Setup (Flax)
# -------------------------------------------------------------------

mask_prob = 0.0
adaptive_tau_mem_mean = 20.
adaptive_tau_mem_std = 5.
adaptive_tau_adp_mean = 150.
adaptive_tau_adp_std = 10.
out_adaptive_tau_mem_mean = 20.
out_adaptive_tau_mem_std = 5.

label_last = False
sub_seq_length = 10
hidden_bias = True
output_bias = True

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
    label_last=label_last,
    sub_seq_length=sub_seq_length,
    hidden_bias=hidden_bias,
    output_bias=output_bias
)

def to_jax(batch):
    inputs, targets = batch
    inputs = jax.device_put(jnp.array(inputs.numpy()))
    targets = jax.device_put(jnp.array(targets.numpy()))
    return jnp.transpose(inputs, (1, 0, 2)), targets # Transpose inputs, keep targets as is

@jax.jit
def nll_loss_fn(logits, labels):
    log_probs = jax.nn.log_softmax(logits)
    # Using one_hot for labels and then summing element-wise for NLLLoss
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    return -jnp.sum(log_probs * one_hot_labels, axis=-1)

@jax.jit
def apply_seq_loss_jax(outputs, target, label_last, sub_seq_length, sequence_length):
    if label_last:
        # For label_last, target is a single class label per sequence
        # outputs are (sequence_length, batch_size, num_classes)
        # We need the last output's loss
        loss = jax.vmap(nll_loss_fn, in_axes=(0, None))(outputs[-1, :, :], target)
        return jnp.mean(loss)
    else:
        # For sequence-wise loss, target is applied to outputs after sub_seq_length
        # target is (batch_size, num_classes)
        # outputs are (sequence_length, batch_size, num_classes)
        outputs_sliced = outputs[sub_seq_length:, :, :]
        # Repeat target along the sequence dimension for applying loss at each step
        target_expanded = jnp.expand_dims(target, axis=0) # (1, batch_size, num_classes)
        target_repeated = jnp.repeat(target_expanded, outputs_sliced.shape[0], axis=0) # (sliced_seq_len, batch_size, num_classes)

        # Apply NLL loss for each element in the sequence
        losses_per_element = jax.vmap(jax.vmap(nll_loss_fn, in_axes=(0, None)))(outputs_sliced, jnp.argmax(target_repeated, axis=-1))
        # Sum over the sequence and batch, then normalize
        return jnp.sum(losses_per_element) / (outputs_sliced.shape[0] * outputs_sliced.shape[1])

@jax.jit
def count_correct_predictions_jax(predictions, targets):
    predicted_classes = jnp.argmax(predictions, axis=-1)
    true_classes = jnp.argmax(targets, axis=-1) # Assuming targets are one-hot encoded for consistency
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

@partial(jax.jit, static_argnames=('label_last', 'sub_seq_length', 'sequence_length'))
def train_step(state, batch, label_last, sub_seq_length, sequence_length):
    inputs, targets = batch

    def loss_fn(params):
        outputs, _, _ = state.apply_fn({'params': params}, inputs)
        loss = apply_seq_loss_jax(outputs, targets, label_last, sub_seq_length, sequence_length)
        return loss, outputs

    (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    # For accuracy calculation, always use the mean of outputs over time, and compare with the single target
    correct = count_correct_predictions_jax(outputs.mean(axis=0), targets)
    accuracy = (correct / targets.shape[0]) * 100.0 # targets.shape[0] is batch_size

    return state, loss, accuracy, outputs.mean(axis=0)

@partial(jax.jit, static_argnames=('label_last', 'sub_seq_length', 'sequence_length'))
def eval_step(state, batch, label_last, sub_seq_length, sequence_length):
    inputs, targets = batch
    outputs, _, _ = state.apply_fn({'params': state.params}, inputs)
    loss = apply_seq_loss_jax(outputs, targets, label_last, sub_seq_length, sequence_length)

    correct = count_correct_predictions_jax(outputs.mean(axis=0), targets)
    accuracy = (correct / targets.shape[0]) * 100.0
    return loss, accuracy

# -------------------------------------------------------------------
# Learning Rate Scheduling
# -------------------------------------------------------------------

optimizer_lr = 0.075
epochs_num = 20
total_steps = len(train_loader)

def create_optimizer(epochs_num, total_steps, initial_lr):
    lr_schedule = optax.linear_schedule(
        init_value=initial_lr,
        end_value=0.0,
        transition_steps=epochs_num * total_steps
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0), # Equivalent to gradient_clip_value
        optax.adam(learning_rate=lr_schedule)
    )

# -------------------------------------------------------------------
# Initialize Model
# -------------------------------------------------------------------

def init_model():
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((sequence_length, batch_size, input_size))
    variables = model.init(rng, dummy_input)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=create_optimizer(epochs_num, total_steps, optimizer_lr),
        batch_stats=variables.get('batch_stats', {}),
        key=rng
    )

# -------------------------------------------------------------------
# Training Loop Functions
# -------------------------------------------------------------------

def train_epoch(state, train_loader, writer, epoch_idx, print_every, label_last, sub_seq_length, sequence_length):
    print_train_loss = 0
    print_correct = 0
    print_total = 0
    end_training_flag = False

    epoch_start_time = time.time()

    for i, batch in enumerate(train_loader):
        batch = to_jax(batch)
        state, loss, acc, batch_outputs_mean = train_step(state, batch, label_last, sub_seq_length, sequence_length)

        loss_val_item = loss.item() # Get Python float for logging

        writer.add_scalar("Loss/train", loss_val_item, epoch_idx * len(train_loader) + i)
        writer.add_scalar("Accuracy/train", acc.item(), epoch_idx * len(train_loader) + i)

        print_train_loss += loss_val_item
        print_total += batch[1].shape[0] # batch_size
        print_correct += count_correct_predictions_jax(batch_outputs_mean, batch[1]) # batch[1] is targets

        if math.isnan(loss_val_item):
            end_training_flag = True
            break

        if i % print_every == (print_every - 1):
            print_acc = (print_correct / print_total) * 100.0

            print(
                "Epoch [{:4d}/{:4d}]  |  Step [{:4d}/{:4d}]  |  Loss/train: {:.6f}, Accuracy/train: {:8.4f}".format(
                    epoch_idx + 1, epochs_num, i + 1, total_steps, print_train_loss / print_every, print_acc), flush=True
            )
            print_correct = 0
            print_total = 0
            print_train_loss = 0

    epoch_end_time = time.time()
    writer.add_scalar("Time/train_epoch_step", (epoch_end_time - epoch_start_time), epoch_idx)

    return state, end_training_flag

def evaluate(state, loader, dataset_size, writer, epoch, prefix, label_last, sub_seq_length, sequence_length):
    total_loss = 0
    total_correct = 0

    for i, batch in enumerate(loader):
        inputs, targets = to_jax(batch)
        loss, acc = eval_step(state, (inputs, targets), label_last, sub_seq_length, sequence_length)

        # Calculate loss value based on label_last or sequence_length
        if label_last:
            loss_value = loss.item()
        else:
            loss_value = loss.item() / (sequence_length - sub_seq_length)

        total_loss += loss_value
        # For evaluation, 'acc' from eval_step already gives batch accuracy.
        # We need to accumulate correct predictions to get overall accuracy.
        # (acc / 100.0) * batch_size = correct predictions in this batch
        total_correct += int(acc / 100.0 * batch[0].shape[0]) # batch[0].shape[0] is original torch batch size

    avg_loss = total_loss / len(loader)
    avg_accuracy = (total_correct / dataset_size) * 100.0

    writer.add_scalar(f"Loss/{prefix}", avg_loss, epoch)
    writer.add_scalar(f"Accuracy/{prefix}", avg_accuracy, epoch)

    return avg_loss, avg_accuracy

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
    best_state_params = state.params
    min_val_loss = float("inf")
    min_val_epoch = 0

    # TensorBoard setup
    rand_num = random.randint(1, 10000)
    opt_str = "{}_Adam({:.2f}),NLL,LinearLR,ll({}),no_gc".format(rand_num, optimizer_lr, label_last)
    net_str = "RSNN(700,128,20,sub_seq({}),bs_{},ep_{}_bias)".format(sub_seq_length, batch_size, epochs_num)
    unit_str = "ALIF(tau_m({},{}),tau_a({},{}),linMask_{})LI(tau_m({},{}))".format(
        adaptive_tau_mem_mean, adaptive_tau_mem_std, adaptive_tau_adp_mean, adaptive_tau_adp_std, mask_prob,
        out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std)
    comment = opt_str + "," + net_str + "," + unit_str

    writer = SummaryWriter(comment=comment)
    log_dir = writer.log_dir
    print(f"TensorBoard log directory: {log_dir}")
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6010"])

    start_time_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    print(start_time_str, comment)

    save_model_params(state.params, f"models/{start_time_str}_{comment}_init.msgpack")

    print(f"Initial model parameters:\n{jax.tree_map(lambda x: x.shape, state.params)}")

    # Record the start time of the entire training run
    training_start_time = time.time()

    # Initial evaluation before training (Epoch 0)
    val_loss, val_acc = evaluate(state, val_loader, val_dataset_size, writer, 0, "val", label_last, sub_seq_length, sequence_length)
    test_loss, test_acc = evaluate(state, test_loader, test_dataset_size, writer, 0, "test", label_last, sub_seq_length, sequence_length)

    print(f"Epoch {0:4d}/{epochs_num}  |  Summary  |  Loss/val: {val_loss:.6f}, Accuracy/val: {val_acc:.4f}%  |  Loss/test: {test_loss:.6f}, "
          f"Accuracy/test: {test_acc:.4f}", flush=True)

    writer.flush()

    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        best_state_params = state.params
        min_val_epoch = 0
        save_model_params(best_state_params, f"models/{start_time_str}_{comment}_best_val_loss_{min_val_loss:.6f}.msgpack")


    end_training = False
    print_every = 115 # From original Torch code

    for epoch in range(epochs_num + 1):
        if epoch < epochs_num: # Run training for epochs 0 to epochs_num - 1
            state, end_training = train_epoch(state, train_loader, writer, epoch, print_every, label_last, sub_seq_length, sequence_length)

            writer.flush()

        # Always evaluate at the end of each epoch (including epoch `epochs_num` for final evaluation)
        val_loss, val_acc = evaluate(state, val_loader, val_dataset_size, writer, epoch, "val", label_last, sub_seq_length, sequence_length)
        test_loss, test_acc = evaluate(state, test_loader, test_dataset_size, writer, epoch, "test", label_last, sub_seq_length, sequence_length)

        print(
            "Epoch [{:4d}/{:4d}]  |  Summary  |  Loss/val: {val_loss:.6f}, Accuracy/val: {val_acc:.4f}%  |  Loss/test: {test_loss:.6f}, "
            "Accuracy/test: {test_acc:.4f}".format(
                epoch, epochs_num, val_loss, val_acc, test_loss, test_acc), flush=True
        )

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            best_state_params = state.params
            save_model_params(best_state_params, f"models/{start_time_str}_{comment}_best_val_loss_{min_val_loss:.6f}.msgpack")

        writer.flush()

        if end_training or jnp.isnan(val_loss):
            print("NaN loss detected or training flagged to end. Ending training.")
            break

    writer.close()

    total_training_time = time.time() - training_start_time
    print(f"{total_training_time:.2f} seconds")
    print("Minimum val loss: {:.6f} at epoch: {}".format(min_val_loss, min_val_epoch))
    return best_state_params, min_val_loss, min_val_epoch


# Run training
final_best_params, final_min_val_loss, final_min_val_epoch = train()
