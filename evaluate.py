import torch
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from conversion.ann_to_snn import convert_to_snn

from utils.energy import (
    calculate_ann_energy,
    calculate_snn_energy
)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root='./datasets',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False
)

# load models
ann_model, snn_model = convert_to_snn(
    "./saved_models/ann_model.pth"
)

ann_model = ann_model.to(device)
snn_model = snn_model.to(device)

# evaluation
ann_correct = 0
snn_correct = 0

total_samples = 0

total_spikes = 0

# ANN MACs
macs_per_image = (784 * 128) + (128 * 10)
total_macs = 0

# timing
ann_start = time.time()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        total_samples += images.size(0)

        total_macs += macs_per_image * images.size(0)

        # ANN
        ann_output = ann_model(images)

        _, ann_pred = ann_output.max(1)

        ann_correct += (ann_pred == labels).sum().item()

ann_end = time.time()

# SNN timing
snn_start = time.time()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # SNN
        snn_output, spikes = snn_model(images)

        _, snn_pred = snn_output.max(1)

        snn_correct += (snn_pred == labels).sum().item()

        total_spikes += spikes

snn_end = time.time()

# accuracy
ann_acc = ann_correct / total_samples * 100
snn_acc = snn_correct / total_samples * 100

# energy
ann_energy = calculate_ann_energy(total_macs)

snn_energy = calculate_snn_energy(total_spikes)

# latency
ann_latency = ann_end - ann_start
snn_latency = snn_end - snn_start

# result
print("\n========== RESULT ==========")

print(f"ANN Accuracy : {ann_acc:.2f}%")
print(f"SNN Accuracy : {snn_acc:.2f}%")

print()

print(f"ANN Energy : {ann_energy:,.0f} pJ")
print(f"SNN Energy : {snn_energy:,.0f} pJ")

print()

print(f"ANN Latency : {ann_latency:.4f} sec")
print(f"SNN Latency : {snn_latency:.4f} sec")

print()

print(f"SNN Energy Ratio : {snn_energy / ann_energy * 100:.4f}%")

print("============================")

from utils.visualization import (
    plot_accuracy,
    plot_energy
)

plot_accuracy(ann_acc, snn_acc)

plot_energy(ann_energy, snn_energy)

print("Charts saved in ./results/")
