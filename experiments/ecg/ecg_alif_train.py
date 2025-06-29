import subprocess

import torch
import torch.nn
import time
from torch.utils.data import DataLoader, random_split
import scipy
import math
from datetime import datetime

import tools
import sys
sys.path.append("../..")
import snn

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import random

###################################################################
# General Settings
###################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = 0

print(device)

####################################################################
# Process Dataset
####################################################################

# TRAIN DATASET #
# check file exists
preprocessed_train_dataset = scipy.io.loadmat('data/QTDB_train.mat')
whole_train_dataset = tools.convert_data_format(preprocessed_train_dataset)

# 618 sequences in whole training dataset
total_train_dataset_size = len(whole_train_dataset)

# 10 % of training data used for validation -> 61
val_dataset_size = int(total_train_dataset_size * 0.1)

# 557 sequences used for training
train_dataset_size = total_train_dataset_size - val_dataset_size

# split whole train dataset randomly
train_dataset, val_dataset = random_split(
    dataset=whole_train_dataset,
    lengths=[train_dataset_size, val_dataset_size]
)

# TEST DATASET #
preprocessed_test_dataset = scipy.io.loadmat('data/QTDB_test.mat')
test_dataset = tools.convert_data_format(preprocessed_test_dataset)

# 141 sequences in test dataset
test_dataset_size = len(test_dataset)

####################################################################
# DataLoader
####################################################################

sequence_length = 1300
input_size = 4
hidden_size = 36
num_classes = 6

train_batch_size = 4
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

####################################################################
# Model setup
####################################################################

# recorded into comment
# fraction of the elements in the hidden.linear.weight to be zero
mask_prob = 0.0

# ALIF alpha tau_mem init normal dist.
adaptive_tau_mem_mean = 20.
adaptive_tau_mem_std = .5

# ALIF rho tau_adp init normal dist.
adaptive_tau_adp_mean = 7.
adaptive_tau_adp_std = .2

# LI alpha tau_mem init normal distribution
out_adaptive_tau_mem_mean = 20.
out_adaptive_tau_mem_std = .5

sub_seq_length = 10

hidden_bias = True
output_bias = True

model = snn.models.SimpleALIFRNN(
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
).to(device)

####################################################################
# Setup experiment (optimizer etc.)
####################################################################

# prevent overwriting in slurm
rand_num = random.randint(1, 10000)

criterion = torch.nn.NLLLoss()

optimizer_lr = 0.05

optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)

# Number of iterations per epoch
total_train_steps = len(train_loader)
total_val_steps = len(val_loader)
total_test_steps = len(test_loader)

epochs_num = 400

# learning rate scheduling
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch_count: 1. - epoch_count / epochs_num)

# [logging] Only thing manually changed in the string: Optimizer, criterion and scheduler!
opt_str = "{}_Adam({}),NLL,LinearLR,no_gc".format(rand_num, optimizer_lr)
net_str = "RSNN(4,36,6,sub_seq_{},bs_{},ep_{},h_o_bias(True))"\
    .format(sub_seq_length, train_batch_size, epochs_num)
unit_str = "ALIF(tau_m({},{}),tau_a({},{}),linMask_{})LI(tau_m({},{}))"\
    .format(adaptive_tau_mem_mean, adaptive_tau_mem_std, adaptive_tau_adp_mean, adaptive_tau_adp_std, mask_prob,
            out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std)

comment = opt_str + "," + net_str + "," + unit_str

writer = SummaryWriter(comment=comment)
log_dir = writer.log_dir  # This gets the actual directory where logs are being saved
subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6008"])

start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
print(start_time, comment)

print(model.state_dict())

save_path = "models/{}_".format(start_time) + comment + ".pt"
save_init_path = "models/{}_init_".format(start_time) + comment + ".pt"

# save initial parameters for analysis
torch.save({'model_state_dict': model.state_dict()}, save_init_path)

min_val_loss = float('inf')

# Dummy init of loss_value for val.
loss_value = 1.

iteration = 0
end_training = False

for epoch in range(epochs_num + 1):

    # check initial performance without training (for plotting purposes)
    # Go into eval mode
    model.eval()

    with torch.no_grad():

        # VALIDATION #

        val_loss = 0
        val_correct = 0

        for i, (inputs, targets) in enumerate(val_loader):

            inputs = inputs.permute(1, 0, 2).to(device)
            targets = targets.permute(1, 0, 2).to(device)

            outputs, _, _ = model(inputs)

            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, targets=targets[sub_seq_length:, :, :])
            val_loss_value = loss.item() / (sequence_length - sub_seq_length)

            val_loss += val_loss_value

            val_correct += tools.count_correct_prediction(predictions=outputs, targets=targets[sub_seq_length:, :, :])

        val_loss /= total_val_steps
        val_acc = (val_correct / (val_dataset_size * (sequence_length - sub_seq_length))) * 100.0

        # Log current val loss and accuracy
        writer.add_scalar(
            "Loss/val",
            val_loss,
            epoch
        )
        writer.add_scalar(
            "accuracy/val",
            val_acc,
            epoch
        )

        # save current best model.
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            saved_best_model = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }, save_path)

        # TEST #

        test_loss = 0
        test_correct = 0

        for i, (inputs, targets) in enumerate(test_loader):

            inputs = inputs.permute(1, 0, 2).to(device)
            targets = targets.permute(1, 0, 2).to(device)

            outputs, _, _ = model(inputs)

            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, targets=targets[sub_seq_length:, :, :])
            test_loss_value = loss.item() / (sequence_length - sub_seq_length)

            test_loss += test_loss_value

            test_correct += tools.count_correct_prediction(predictions=outputs, targets=targets[sub_seq_length:, :, :])

        test_loss /= total_test_steps
        test_acc = (test_correct / (test_dataset_size * (sequence_length - sub_seq_length))) * 100.0



        # Log current test loss and accuracy
        writer.add_scalar(
            "Loss/test",
            test_loss,
            epoch
        )
        writer.add_scalar(
            "accuracy/test",
            test_acc,
            epoch
        )

    print(
        "Epoch [{:4d}/{:4d}]  |  Summary | Loss/val: {:.6f}, Accuracy/val: {:8.4f}  | "
        " Loss/test: {:.6f}, Accuracy/test: {:8.4f}"
        .format(epoch, epochs_num, val_loss, val_acc, test_loss, test_acc),
        flush=True
    )

    # Update logging outputs
    writer.flush()

    # TRAIN #

    # run training from 0 to 399 epochs
    if epoch < epochs_num:

        print_train_loss = 0
        print_train_correct = 0

        # go to training mode
        model.train()


        epoch_start_time = time.time()
        for i, (inputs, targets) in enumerate(train_loader):

            current_batch_size = len(inputs)

            # reshape inputs to (sequence_length, batch_size, input_size)
            inputs = inputs.permute(1, 0, 2).to(device)

            # reshape targets to (sequence_length, batch_size, num_classes)
            targets = targets.permute(1, 0, 2).to(device)

            # clear gradients
            optimizer.zero_grad()

            outputs, _ , _= model(inputs)

            # accumulate loss for each time step and batch
            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, targets=targets[sub_seq_length:, :, :])
            loss_value = loss.item() / (sequence_length - sub_seq_length)

            # calculate the gradients
            loss.backward()

            # perform learning step
            optimizer.step()

            # sum up loss_value for each iteration
            print_train_loss += loss_value

            # calculate batch accuracy
            batch_correct = tools.count_correct_prediction(predictions=outputs, targets=targets[sub_seq_length:, :, :])
            print_train_correct += batch_correct

            batch_accuracy = (batch_correct / (current_batch_size * (sequence_length - sub_seq_length))) * 100.0

            # Log current loss and accuracy
            writer.add_scalar(
                "Loss/train",
                loss_value,
                iteration
            )
            writer.add_scalar(
                "accuracy/train",
                batch_accuracy,
                iteration
            )

            if math.isnan(loss_value):
                end_training = True
                break

            iteration += 1

        epoch_end_time = time.time()

        writer.add_scalar("Time/train_epoch_step", (epoch_end_time - epoch_start_time), epoch)


        print_train_loss /= total_train_steps
        print_acc = (print_train_correct / (train_dataset_size * (sequence_length - sub_seq_length))) * 100.0

        print(
            "Epoch [{:4d}/{:4d}]  | Loss/train: {:.6f}, Accuracy/train: {:8.4f}"
            .format(epoch + 1, epochs_num, print_train_loss, print_acc), flush=True
        )

        # Update logging outputs
        writer.flush()

        # Apply learning rate scheduling
        scheduler.step()

    if end_training:
        break

print('Min val loss: {:.6f} at epoch {}'.format(min_val_loss, min_val_epoch))
# print(saved_best_model)






