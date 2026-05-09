import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
import torch
import time
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from conversion.ann_to_snn import convert_to_snn
from utils.energy import calculate_snn_energy

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

# load model
_, snn_model = convert_to_snn(
    "./saved_models/ann_model.pth"
)

snn_model = snn_model.to(device)

# experiment settings
timesteps = [5, 10, 20, 50]

accuracy_results = []
energy_results = []
latency_results = []

for steps in timesteps:

    print(f"\nTesting timestep = {steps}")

    correct = 0
    total = 0
    total_spikes = 0

    start_time = time.time()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs, spikes = snn_model(
                images,
                num_steps=steps
            )

            _, predicted = outputs.max(1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

            total_spikes += spikes

    end_time = time.time()

    accuracy = correct / total * 100

    energy = calculate_snn_energy(total_spikes)

    latency = end_time - start_time

    accuracy_results.append(accuracy)
    energy_results.append(energy)
    latency_results.append(latency)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Energy: {energy:,.0f} pJ")
    print(f"Latency: {latency:.4f} sec")

# plot accuracy
plt.figure(figsize=(7, 5))

plt.plot(timesteps, accuracy_results, marker='o')

plt.xlabel("Timesteps")
plt.ylabel("Accuracy (%)")

plt.title("Timesteps vs Accuracy")

plt.grid(True)

plt.savefig("./results/timestep_accuracy.png")

# plot energy
plt.figure(figsize=(7, 5))

plt.plot(timesteps, energy_results, marker='o')

plt.xlabel("Timesteps")
plt.ylabel("Energy (pJ)")

plt.title("Timesteps vs Energy")

plt.yscale('log')

plt.grid(True)

plt.savefig("./results/timestep_energy.png")

# plot latency
plt.figure(figsize=(7, 5))

plt.plot(timesteps, latency_results, marker='o')

plt.xlabel("Timesteps")
plt.ylabel("Latency (sec)")

plt.title("Timesteps vs Latency")

plt.grid(True)

plt.savefig("./results/timestep_latency.png")

print("\nExperiment completed.")
