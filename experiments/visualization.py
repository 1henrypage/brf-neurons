import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

current_dir = os.getcwd()

def load_model_data(folder, model_name):
    csv_files = glob.glob(folder)
    all_runs = []

    for file in csv_files:
        df = pd.read_csv(file)
        accuracy_df = df[df["metric"].isin(["Accuracy/test", "accuracy/test"])]
        accuracy_df = accuracy_df[["step", "value"]].copy()
        accuracy_df["model"] = model_name
        all_runs.append(accuracy_df)

    return pd.concat(all_runs, ignore_index=True)

# Paths for datasets
model_paths_ecg = {
    "BRF model": os.path.join(current_dir, "ecg/csv_files/ecg_best_model_run_*.csv"),
    "RF model": os.path.join(current_dir, "ecg/csv_files/ecg_vrf-NoR_model_run_*.csv"),
    "ALIF model": os.path.join(current_dir, "ecg/csv_files/ecg_alif_best_model_run_*.csv")
}

model_paths_shd = {
    "BRF model": os.path.join(current_dir, "SHD/csv_files/shd_best_model_run_*.csv"),
    "RF model": os.path.join(current_dir, "SHD/csv_files/shd_vrf-NoR_model_run_*.csv"),
    "ALIF model": os.path.join(current_dir, "SHD/csv_files/shd_alif_model_run_*.csv")
}

model_paths_smnist = {
    "BRF model": os.path.join(current_dir, "smnist/csv_files/smnist_ll_model_run_*.csv"),
    "RF model": os.path.join(current_dir, "smnist/csv_files/smnist_no_variant_run_*.csv"),
    "ALIF model": os.path.join(current_dir, "smnist/csv_files/smnist_alif_tbptt_ll_model_run_*.csv")
}

model_paths_psmnist = {
    "BRF model": os.path.join(current_dir, "smnist/csv_files/psmnist_model_run_*.csv"),
    "ALIF model": os.path.join(current_dir, "smnist/csv_files/psmnist_alif_tbptt_model_run_*.csv")
}

all_data_ecg = pd.concat([
    load_model_data(folder, model_name)
    for model_name, folder in model_paths_ecg.items()
], ignore_index=True)

all_data_shd = pd.concat([
    load_model_data(folder, model_name)
    for model_name, folder in model_paths_shd.items()
], ignore_index=True)

all_data_smnist = pd.concat([
    load_model_data(folder, model_name)
    for model_name, folder in model_paths_smnist.items()
], ignore_index=True)

all_data_psmnist = pd.concat([
    load_model_data(folder, model_name)
    for model_name, folder in model_paths_psmnist.items()
], ignore_index=True)

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

# S-MNIST plot
sns.lineplot(data=all_data_smnist, x="step", y="value", hue="model", ax=axes[0])
axes[0].set_xlabel("Epoch", fontsize=16)
axes[0].set_ylabel("")
axes[0].legend(title="S-MNIST", fontsize=16, title_fontsize='18', loc='best')
axes[0].grid(True)

# PS-MNIST plot
sns.lineplot(data=all_data_psmnist, x="step", y="value", hue="model", ax=axes[1])
axes[1].set_xlabel("Epoch", fontsize=16)
axes[1].set_ylabel("")
axes[1].legend(title="PS-MNIST", fontsize=16, title_fontsize='18', loc='best')
axes[1].grid(True)

# ECG plot
sns.lineplot(data=all_data_ecg, x="step", y="value", hue="model", ax=axes[2])
axes[2].set_xlabel("Epoch", fontsize=16)
axes[2].set_ylabel("")
axes[2].legend(title="ECG", fontsize=16, title_fontsize='18', loc='best')
axes[2].grid(True)

# SHD plot
sns.lineplot(data=all_data_shd, x="step", y="value", hue="model", ax=axes[3])
axes[3].set_xlabel("Epoch", fontsize=16)
axes[3].set_ylabel("")
axes[3].legend(title="SHD", fontsize=16, title_fontsize='18', loc='best')
axes[3].grid(True)

fig.text(0.04, 0.5, 'Test Accuracy', va='center', rotation='vertical', fontsize=16)

plt.tight_layout()
plt.subplots_adjust(left=0.08)
plt.savefig("model_comparison.png", dpi=300)
plt.show()
