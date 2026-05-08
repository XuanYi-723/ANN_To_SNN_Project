import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.ann_model import NetANN

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./datasets',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# model
model = NetANN().to(device)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
epochs = 5

print("開始訓練 ANN...")

for epoch in range(epochs):

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)

        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# save model
torch.save(model.state_dict(), "./saved_models/ann_model.pth")

print("ANN 模型已儲存")
